
 
# Real Estate Prediction

This project aims to build and evaluate multiple regression models to predict real estate prices based on various features. The project uses a dataset with features such as 'Age' and 'Salary' and includes models like Linear Regression, Decision Tree, and Random Forest.

## Project Structure

The project is organized as follows:

- `main.py`: Entry point of the application. Orchestrates data loading, model training, evaluation, and prediction processes.
- `models/train.py`: Contains functions for training different types of regression models.
- `models/evaluate.py`: Contains functions for evaluating the trained models.
- `models/predict.py`: Contains functions for loading a saved model and making predictions.
- `utils/logger.py`: Sets up logging for the application.
- `utils/helpers.py`: Contains helper functions, such as loading data.

## Requirements

Ensure you have the following Python packages installed:
- `pandas`
- `scikit-learn`
- `pickle`

You can install these dependencies using pip:
```bash
pip install pandas scikit-learn
