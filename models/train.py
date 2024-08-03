from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def train_linear_regression(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=1234)
    model = LinearRegression().fit(X_train, y_train)
    logger.info("Linear Regression model trained.")
    return model

def train_decision_tree(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=1234)
    model = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=567).fit(X_train, y_train)
    logger.info("Decision Tree model trained.")
    return model

def train_random_forest(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=1234)
    model = RandomForestRegressor(n_estimators=200, criterion='absolute_error').fit(X_train, y_train)
    logger.info("Random Forest model trained.")
    return model
