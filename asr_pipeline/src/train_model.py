import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from src.config.config import DEVICE, BATCH_SIZE, EPOCHS, MODEL_PATH, EMBEDDING_DIM
from src.utils.logger import get_logger

logger = get_logger("ModelTraining")

class SemanticMLPClassifier(nn.Module):
    def __init__(self, input_dim=EMBEDDING_DIM, num_classes=5):
        """
        Lightweight MLP Classifier as defined in the architectural layer format.
        """
        super(SemanticMLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
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

    model = SemanticMLPClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
                
        val_loss_epoch = val_loss / len(val_loader)
        
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

    logger.info(f"Model saved at {MODEL_PATH}")
    return model