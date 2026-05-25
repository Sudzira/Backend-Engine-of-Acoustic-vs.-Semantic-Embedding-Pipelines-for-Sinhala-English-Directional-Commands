from src.data_preprocessing import preprocess_run
from src.feature_engineering import extract_features
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.utils.logger import get_logger

logger = get_logger('AcousticPipeline')

def main():
    logger.info('Starting Directional Command Recognition Pipeline')
    
    # Step 1: Preprocessing
    df = preprocess_run()
    logger.info('Data preprocessing completed')
    
    # Step 2: Feature Extraction (DistilHuBERT embeddings)
    train_data, val_data = extract_features(df)
    logger.info('Feature extraction and train/val split completed')
    
    # Step 3: Model Training
    if train_data and val_data:
        model = train_model(train_data, val_data)
        logger.info('Model training completed')
        
        # Step 4: Evaluation
        evaluate_model(model, val_data) # Evaluating on val set for demonstration
        logger.info('Model evaluation completed')
        
    logger.info('Pipeline completed successfully')

if __name__ == '__main__':
    main()