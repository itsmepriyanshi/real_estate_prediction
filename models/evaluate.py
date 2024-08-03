from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, df, target_column, model_name):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    logger.info(f"{model_name} Train MAE: {train_mae}")
    logger.info(f"{model_name} Test MAE: {test_mae}")