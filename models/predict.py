import pickle
import logging

logger = logging.getLogger(__name__)

def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise

def make_prediction(model, data):
    import pandas as pd
    # Ensure the data is a DataFrame with the correct column names
    df = pd.DataFrame(data, columns=['Age'])  # Adjust columns based on the training features
    prediction = model.predict(df)
    return prediction
