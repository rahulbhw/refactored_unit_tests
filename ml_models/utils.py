import pandas as pd
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import platform, socket, re, uuid, json, psutil, logging
from platform import python_version
import numpy as np



def get_data():
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer["data"], columns=cancer["feature_names"])
    df["target"] = cancer["target"]
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101
    )
    return X_train, X_test, y_train, y_test


def get_system_info():
    try:
        info = {}
        info["platform"] = platform.system()
        info["platform-release"] = platform.release()
        info["platform-version"] = platform.version()
        info["architecture"] = platform.machine()
        info["processor"] = platform.processor()
        info["ram"] = str(round(psutil.virtual_memory().total / (1024.0**3))) + " GB"
        info["cores"] = psutil.cpu_count(logical=False)
        info["threads"] = int(psutil.cpu_count() / psutil.cpu_count(logical=False))
        info["python-version"] = python_version()
        return json.dumps(info)
    except Exception as e:
        logging.exception(e)


def get_np_configs():
    return np.show_config()


def get_mkl_info():
    try:
        import mkl
        return mkl.get_version_string()
    except ImportError:
        return "mkl not installed"
