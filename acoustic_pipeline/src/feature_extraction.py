import os
import torch
from transformers import AutoProcessor, AutoModel
from sklearn.model_selection import train_test_split
from src.config.config import DEVICE, FEATURE_OUTPUT_PATH
from src.utils.logger import get_logger

logger = get_logger("FeatureEngineering")

def extract_features(df):
    # This acts as a placeholder for the DistilHuBERT feature extraction
    # that transforms the processed .wav files into pt tensors.
    # We simulate loading the tensors from disk as per your notebook logic.
    os.makedirs(FEATURE_OUTPUT_PATH, exist_ok=True)
    
    train_path = os.path.join(FEATURE_OUTPUT_PATH, "train_features.pt")
    val_path = os.path.join(FEATURE_OUTPUT_PATH, "val_features.pt")
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        train_data = torch.load(train_path)
        val_data = torch.load(val_path)
        return train_data, val_data
    else:
        logger.error("Features not found. Please ensure feature extraction was performed.")
        return None, None