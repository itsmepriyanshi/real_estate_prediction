import logging
from utils.logger import setup_logging
from models.train import train_linear_regression, train_decision_tree, train_random_forest
from models.evaluate import evaluate_model
from models.predict import load_model, make_prediction
from utils.helpers import load_data
import os
import pickle

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
def main():
    logger.info("Starting the model training and evaluation process.")
    
    # Verify the current working directory
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Load data
    try:
        df = load_data('data/age_salary.csv')
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return
    
    # Print column names for debugging
    logger.info(f"Loaded columns: {df.columns}")
    
    # Update column names and target variable
    target_column = 'Salary'  # Update this to the column you want to predict
    if target_column not in df.columns:
        logger.error(f"'{target_column}' column not found in the dataset.")
        return
    
    # Train models
    lr_model = train_linear_regression(df, target_column)
    dt_model = train_decision_tree(df, target_column)
    rf_model = train_random_forest(df, target_column)
    
    # Evaluate models
    evaluate_model(lr_model, df, target_column, model_name="Linear Regression")
    evaluate_model(dt_model, df, target_column, model_name="Decision Tree")
    evaluate_model(rf_model, df, target_column, model_name="Random Forest")
    
    # Save the best model (assuming Random Forest is the best based on evaluation)
    model_path = 'models/RE_Model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)
    
    # Load the model and make a prediction
    loaded_model = load_model(model_path)
    sample_data = [[35]]  # Example input data with only the features used in training (e.g., Age)
    prediction = make_prediction(loaded_model, sample_data)
    logger.info(f"Predicted {target_column}: {prediction}")

if __name__ == "__main__":
    main()
