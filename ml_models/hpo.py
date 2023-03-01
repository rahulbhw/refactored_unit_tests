from utils import get_data, get_system_info, get_np_configs, get_mkl_info
from config import get_grid, get_model
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass
from sklearn.model_selection import GridSearchCV
import time
import joblib
import psutil
from dask.distributed import Client
import ipyparallel as ipp
from ipyparallel.joblib import IPythonParallelBackend
try:
    from ray.util.joblib import register_ray
except ImportError:
    pass
from joblibspark import register_spark
try:
    from tune_sklearn import TuneGridSearchCV
except ImportError:
    pass
import os
from loguru import logger

if __name__ == "__main__":
    env_name = os.environ['CONDA_DEFAULT_ENV']
    logger.add(f"{env_name}.log")
    logger.info(get_system_info())
    logger.info(get_np_configs())
    logger.info(get_mkl_info())
    X_train, X_test, y_train, y_test = get_data()
    logger.info(X_train.shape)
    logger.info(X_test.shape)
    model = get_model()
    parameter_grid = get_grid()
    grid = GridSearchCV(model, parameter_grid, verbose=1)
    ############################## loky, n_jobs=1, CPU ############################
    current_process = psutil.Process()
    subproc_before = set([p.pid for p in current_process.children(recursive=True)])
    start = time.time()
    with joblib.parallel_backend("loky", n_jobs=1):
        grid.fit(X_train, y_train)
        grid_predictions = grid.predict(X_test)
    end = time.time()
    logger.info(f"with context manager: true, backend: loky, n_jobs = 1, time taken: {end - start}")
    subproc_after = set([p.pid for p in current_process.children(recursive=True)])
    for subproc in subproc_after - subproc_before:
        psutil.Process(subproc).terminate()
    # ############################ loky, n_jobs=-1, CPU ############################
    current_process = psutil.Process()
    subproc_before = set([p.pid for p in current_process.children(recursive=True)])
    start = time.time()
    with joblib.parallel_backend("loky", n_jobs=-1):
        grid.fit(X_train, y_train)
        grid_predictions = grid.predict(X_test)
    end = time.time()
    logger.info(f"with context manager: true, backend: loky, n_jobs = -1, time taken: {end - start}")
    subproc_after = set([p.pid for p in current_process.children(recursive=True)])
    for subproc in subproc_after - subproc_before:
        psutil.Process(subproc).terminate()
    # ############################ multiprocessing, n_jobs=-1, CPU ############################
    current_process = psutil.Process()
    subproc_before = set([p.pid for p in current_process.children(recursive=True)])
    start = time.time()
    with joblib.parallel_backend("multiprocessing", n_jobs=-1):
        grid.fit(X_train, y_train)
        grid_predictions = grid.predict(X_test)
    end = time.time()
    logger.info(f"with context manager: true, backend: multiprocessing, n_jobs = -1, time taken: {end - start}")
    subproc_after = set([p.pid for p in current_process.children(recursive=True)])
    for subproc in subproc_after - subproc_before:
        psutil.Process(subproc).terminate()
    # ############################ threading, n_jobs=-1, CPU ############################
    current_process = psutil.Process()
    subproc_before = set([p.pid for p in current_process.children(recursive=True)])
    start = time.time()
    with joblib.parallel_backend("threading", n_jobs=-1):
        grid.fit(X_train, y_train)
        grid_predictions = grid.predict(X_test)
    end = time.time()
    logger.info(f"with context manager: true, backend: threading, n_jobs = -1, time taken: {end - start}")
    subproc_after = set([p.pid for p in current_process.children(recursive=True)])
    for subproc in subproc_after - subproc_before:
        psutil.Process(subproc).terminate()
    # ############################ dask, n_jobs=-1, CPU ############################
    client = Client(processes=False)
    start = time.time()
    with joblib.parallel_backend("dask", n_jobs=-1):
        grid.fit(X_train, y_train)
        grid_predictions = grid.predict(X_test)
    end = time.time()
    logger.info(f"with context manager: true, backend: dask, n_jobs = -1, time taken: {end - start}")
    client.shutdown()
    ############################ ipyparallel, n_jobs=-1, CPU ############################
    with ipp.Cluster() as rc:
        bview = rc.load_balanced_view()
        joblib.register_parallel_backend('ipyparallel', lambda : IPythonParallelBackend(view=bview))
        start = time.time()
        with joblib.parallel_backend("ipyparallel", n_jobs=-1):
            grid.fit(X_train, y_train)
            grid_predictions = grid.predict(X_test)
        end = time.time()
        logger.info(f"with context manager: true, backend: ipyparallel, n_jobs = -1, time taken: {end - start}")
    ############################ ray, n_jobs=-1, CPU ############################
    register_ray()
    start = time.time()
    with joblib.parallel_backend("ray", n_jobs=-1):
        grid.fit(X_train, y_train)
        grid_predictions = grid.predict(X_test)
    end = time.time()
    logger.info(f"with context manager: true, backend: ray, n_jobs = -1, time taken: {end - start}")
    ############################ spark, n_jobs=-1, Spark ############################
    register_spark()
    start = time.time()
    with joblib.parallel_backend("spark", n_jobs=-1):
        grid.fit(X_train, y_train)
        grid_predictions = grid.predict(X_test)
    end = time.time()
    logger.info(f"with context manager: true, backend: spark, n_jobs = -1, time taken: {end - start}")
    ############################ ray, n_jobs=-1, GPU ############################
    gpu_grid = TuneGridSearchCV(model, parameter_grid, verbose=1, use_gpu=True, cv = 30)
    register_ray()
    start = time.time()
    with joblib.parallel_backend("ray"):
        gpu_grid.fit(X_train, y_train)
        gpu_grid_predictions = gpu_grid.predict(X_test)
    end = time.time()
    logger.info(f"with context manager: true, backend: ray, n_jobs = -1, time taken: {end - start}")

