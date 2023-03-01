from utils import get_data, get_grid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from watermark import watermark
import time
import joblib
import ipyparallel as ipp
from ipyparallel.joblib import IPythonParallelBackend

if __name__ == '__main__':
    print(watermark(packages="numpy,scipy,sklearn,pandas,joblib,ray,dask"))
    with ipp.Cluster() as rc:
        print(rc.ids)
        bview = rc.load_balanced_view()
        joblib.register_parallel_backend('ipyparallel', lambda : IPythonParallelBackend(view=bview))
        X_train, X_test, y_train, y_test = get_data()
        param_grid = get_grid()
        rfc = RandomForestClassifier()
        grid = GridSearchCV(rfc, param_grid, verbose=1)
        start = time.time()
        with joblib.parallel_backend('ipyparallel'):
            grid.fit(X_train, y_train)
            grid_predictions = grid.predict(X_test)
        end = time.time()
        print(f"Time taken: {end - start}")
        print(confusion_matrix(y_test, grid_predictions))
        print(classification_report(y_test, grid_predictions))