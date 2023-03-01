try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass
from sklearn.ensemble import RandomForestClassifier


def get_grid():
    param_grid = {
        "n_estimators": [10, 20, 30, 40],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    return param_grid


def get_model():
    model = RandomForestClassifier()
    return model
