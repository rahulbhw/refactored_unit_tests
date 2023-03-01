from utils import get_data, get_grid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from watermark import watermark
import time
import joblib


if __name__ == '__main__':
    print(watermark(packages="numpy,scipy,sklearn,pandas,joblib,ray,dask"))
    X_train, X_test, y_train, y_test = get_data()
    param_grid = get_grid()
    rfc = RandomForestClassifier()
    grid = GridSearchCV(rfc, param_grid, verbose=1)
    start = time.time()
    with joblib.parallel_backend('threading', n_jobs=-1):
        grid.fit(X_train, y_train)
        grid_predictions = grid.predict(X_test)
    end = time.time()
    print(f"Time taken: {end - start}")
    print(confusion_matrix(y_test, grid_predictions))
    print(classification_report(y_test, grid_predictions))