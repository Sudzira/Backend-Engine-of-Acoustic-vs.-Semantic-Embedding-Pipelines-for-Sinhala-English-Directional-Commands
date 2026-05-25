from src.data_preprocessing import preprocess_run
from src.feature_engineering import extract_semantic_features
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.utils.logger import get_logger

logger = get_logger('ASRSemanticPipeline')

def main():
    logger.info('Starting ASR + Semantic Similarity Pipeline')
    
    # Step 1: Preprocessing (Silence trimming, RMS Norm)
    df = preprocess_run()
    logger.info('Data preprocessing completed')
    
    # Step 2: Feature Extraction (Whisper Transcription -> MiniLM Embedding)
    train_data, val_data = extract_semantic_features(df)
    logger.info('Feature extraction and data splitting completed')
    
    # Step 3: Model Training (Lightweight MLP)
    if train_data and val_data:
        model = train_model(train_data, val_data)
        logger.info('Model training completed')
        
        # Step 4: Evaluation
        evaluate_model(model, val_data)
        logger.info('Model evaluation completed')
        
    logger.info('Pipeline completed successfully')

if __name__ == '__main__':
    main()