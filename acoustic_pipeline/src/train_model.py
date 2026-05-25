import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from src.config.config import DEVICE, BATCH_SIZE, EPOCHS, MODEL_PATH
from src.utils.logger import get_logger

logger = get_logger("ModelTraining")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean', label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

class CommandClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=5):
        super(CommandClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(0.20),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.40),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def train_model(train_data, val_data):
    X_train, y_train = train_data['embeddings'], train_data['labels']
    X_val, y_val = val_data['embeddings'], val_data['labels']

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CommandClassifier().to(DEVICE)
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for inputs, labels_batch in train_loader:
            inputs, labels_batch = inputs.to(DEVICE), labels_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels_batch in val_loader:
                inputs, labels_batch = inputs.to(DEVICE), labels_batch.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                
        scheduler.step()
        val_loss_epoch = val_loss / len(val_loader)
        
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

    logger.info(f"Model saved at {MODEL_PATH}")
    return model