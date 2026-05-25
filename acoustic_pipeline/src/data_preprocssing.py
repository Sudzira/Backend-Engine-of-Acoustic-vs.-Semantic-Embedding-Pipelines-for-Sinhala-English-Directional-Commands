import os
import glob
import pandas as pd
import librosa
import soundfile as sf
from src.config.config import DATA_RAW_PATH, DATA_PROCESSED_PATH, LABELS
from src.utils.logger import get_logger

logger = get_logger("DataPreprocessing")

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    y_norm = librosa.util.normalize(y_trimmed)
    return y_norm, sr

def preprocess_run():
    data = []
    for label in LABELS:
        path = os.path.join(DATA_RAW_PATH, label)
        if not os.path.exists(path):
            continue
        files = glob.glob(os.path.join(path, "*.wav"))
        for f in files:
            data.append({"path": f, "label": label})
    
    df = pd.DataFrame(data)
    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
    
    processed_paths = []
    for idx, row in df.iterrows():
        filename = os.path.basename(row['path'])
        label_subdir = os.path.join(DATA_PROCESSED_PATH, row['label'])
        os.makedirs(label_subdir, exist_ok=True)
        new_path = os.path.join(label_subdir, filename)
        
        if not os.path.exists(new_path):
            try:
                y_proc, sr = preprocess_audio(row['path'])
                sf.write(new_path, y_proc, sr)
                processed_paths.append(new_path)
            except Exception as e:
                logger.error(f"Failed to process {row['path']}: {e}")
                processed_paths.append(row['path'])
        else:
            processed_paths.append(new_path)
            
    df['path'] = processed_paths
    return df