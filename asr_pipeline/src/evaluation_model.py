import torch
from sklearn.metrics import classification_report
from src.config.config import DEVICE, LABELS
from src.utils.logger import get_logger

logger = get_logger("ModelEvaluation")

def evaluate_model(model, test_data):
    X_test, y_test = test_data['embeddings'], test_data['labels']
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        
    report = classification_report(y_test.cpu().numpy(), predicted.cpu().numpy(), target_names=LABELS)
    logger.info(f"Classification Report:\n{report}")
    return report