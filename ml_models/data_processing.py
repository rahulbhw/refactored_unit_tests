# %%
try:
    import modin.pandas as pd
    from sklearnex import patch_sklearn
    patch_sklearn()
    import ray
    ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}})
except ImportError:
    import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
def split_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test
# %%
import numpy as np
df = pd.read_csv('regression_data.csv.gzip', compression='gzip')
df['factor'] = np.random.randint(0, 1000, df.shape[0])
df['factor'] = df['factor'].astype('str')
agg_data = df.groupby(['factor'],as_index= False).sum()
agg_data['factor'].unique().tolist()
agg_data.drop_duplicates()
df = df.rename(columns={'factor': 'id'})
df.columns.tolist()
# df.dtypes.apply(lambda x: x.name).to_dict() # `Series.to_dict` is not currently supported by PandasOnRay
# df.isna().any().to_dict()                   # `Series.to_dict` is not currently supported by PandasOnRay
# df.isna().any().to_dict()                   # `Series.to_dict` is not currently supported by PandasOnRay
df.fillna(0)
df["target"].agg(["min", "max"])
# Do a feature extraction and selection and then train a model on the data using sklearn and modin





# %%

def check_columns(df, columns):
    return set(columns).issubset(set(df.columns.tolist()))

def log_missing_columns(df, columns):
    if not check_columns(df, columns):
        print("Columns not present in dataframe", set(columns) - set(df.columns.tolist()))

def select_columns(df, columns):
    log_missing_columns(df, columns)
    return df[columns]

def sort_dict_by_keys(dictionary):
    return {key: dictionary[key] for key in sorted(dictionary.keys())}

def sort_dict_by_values(dictionary):
    return {key: value for key, value in sorted(dictionary.items(), key=lambda item: item[1])}

def get_column_datatypes(df):
    # NOTE: `Series.to_dict` is not currently supported by PandasOnRay
    # Run this on pandas with sample data
    return df.dtypes.apply(lambda x: x.name).to_dict()

def change_datatypes(dictionary):
    return {key: value.replace('64', '') for key, value in dictionary.items()}

def change_values_to_number(dictionary):
    return {key: value.replace('int', 'number').replace('float', 'number') for key, value in dictionary.items()}

def check_datatypes(df, column_datatypes):
    df_column_datatypes = get_column_datatypes(df)
    return df_column_datatypes == column_datatypes

def check_nan_values(df):
    # NOTE: `Series.to_dict` is not currently supported by PandasOnRay
    # Run this on pandas with sample data
    return df.isna().any().to_dict()

def check_nan_values_in_columns(df, columns):
    nan_values = check_nan_values(df)
    return {key: value for key, value in nan_values.items() if key in columns and value == True}

def check_non_nan_values_in_columns(df, columns):
    nan_values = check_nan_values(df)
    return {key: value for key, value in nan_values.items() if key in columns and value == False}

def fill_nan_values(df, fill_values):
    for key, value in fill_values.items():
        df[key] = df[key].fillna(value)
    return df

def check_columns_in_range(df, columns):
    for key, value in columns.items():
        if not df[key].between(value['min'], value['max']).all():
            return False
    return True

def get_unique_values(df, columns):
    return {key: df[key].unique().tolist() for key in columns}

# check all in columns has unique values or not. take a df and a dict as input. where key is column name and value is a list of unique values

def check_unique_values(df, columns):
    unique_values = get_unique_values(df, columns)
    unique_status = {}
    for key, value in unique_values.items():
        if not set(value) == set(columns[key]):
            unique_status[key] = False
        else:
            unique_status[key] = True
    return unique_status

def check_missing_unique_values(df, columns):
    unique_values = get_unique_values(df, columns)
    missing_unique_values = {}
    for key, value in unique_values.items():
        if not set(value) == set(columns[key]):
            missing_unique_values[key] = list(set(columns[key]) - set(value))
        else:
            missing_unique_values[key] = []
    return missing_unique_values

def rename_columns(df, columns_mapper):
    df = df.rename(columns=columns_mapper)
    return df

def check_NaT_values(df, date_dict):
    nat_status = {}
    for key, value in date_dict.items():
        if not pd.to_datetime(df[key], format=value, errors='coerce').notnull().all():
            nat_status[key] = False
        else:
            nat_status[key] = True
    return nat_status

def check_date_columns_in_range(df, date_dict):
    date_status = {}
    for key, value in date_dict.items():
        if not pd.to_datetime(df[key], format=value['format'], errors='coerce').between(value['min'], value['max']).all():
            date_status[key] = False
        else:
            date_status[key] = True
    return date_status

def filter_date_columns_in_range(df, date_dict):
    date_status = check_date_columns_in_range(df, date_dict)
    for key, value in date_status.items():
        if value == False:
            df = df.drop(key, axis=1)
    return df

