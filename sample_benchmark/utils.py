import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def get_data():
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
    df['target'] = cancer['target']
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test

def get_grid():
    param_grid = {'n_estimators': [10, 20, 30, 40],
                'max_depth': [5, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]}
    return param_grid