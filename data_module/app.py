import os
from configs.config import get_config
import pandas as pd
import numpy as np
import json
import joblib
from loguru import logger

# TODO: modify create_folder, create_folders, check_file_exists, check_files_exists for Azure databricks file system


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def create_folders(folder_list):
    for folder in folder_list:
        create_folder(folder)


def check_file_exists(file_path):
    return os.path.exists(file_path)


def check_files_exists(file_list):
    file_status = {}
    for file in file_list:
        if not check_file_exists(file):
            file_status[file] = False
        else:
            file_status[file] = True
    return file_status


def show_config():
    config = get_config()
    for key, value in config.items():
        print(key, value)


def synthetic_data():
    # set seed for reproducibility
    np.random.seed(0)
    hierarchical_columns = [
        "poc_id",
        "sku_id",
        "region_id",
        "country_id",
        "city_id",
        "order_id",
        "channel_id",
        "brand_id",
        "subsegment_id",
        "state_id",
        "segment_id",
        "route_id",
        " delivery_center_id",
        "deliver_region_id",
        "sales_route_id",
    ]
    date_columns = ["date", "join_date", "bees_join_date"]
    integer_columns = ["days_since_last_order", "is_purchased"]
    float_columns = [
        "quantity",
        "average_products_per_order",
        "interaction_score",
        "number_of_orders_for_pid",
        "credit_total",
        "credit_payment_terms",
        "maltas",
        "cervezas",
        "agua",
        "num_skus",
        "num_orders",
        "skus_per_order",
        "average_order_revenue",
    ]
    df = pd.DataFrame(
        columns=hierarchical_columns + date_columns + integer_columns + float_columns
    )
    df[hierarchical_columns] = np.random.randint(
        1, 100, size=(1000, len(hierarchical_columns))
    ).astype(str)
    df[float_columns] = np.random.rand(1000, len(float_columns))
    for col in date_columns:
        df[col] = pd.date_range("1/1/2020", periods=1000, freq="D")
    df[integer_columns] = np.random.randint(1, 100, size=(1000, len(integer_columns)))
    return df


def download_file_from_blob(file_path, blob_path):
    from azure.storage.blob import BlobServiceClient
    from configs.config import get_config

    config = get_config()
    blob_service_client = BlobServiceClient.from_connection_string(
        config["azure"]["connection_string"]
    )
    blob_client = blob_service_client.get_blob_client(
        container=config["azure"]["container_name"], blob=blob_path
    )
    with open(file_path, "wb") as my_blob:
        blob_data = blob_client.download_blob()
        blob_data.readinto(my_blob)


def download_files_from_blob(file_list, blob_list):
    for ix in range(len(file_list)):
        download_file_from_blob(file_list[ix], blob_list[ix])


def upload_file_to_blob(file_path, blob_path):
    from azure.storage.blob import BlobServiceClient
    from configs.config import get_config

    config = get_config()
    blob_service_client = BlobServiceClient.from_connection_string(
        config["azure"]["connection_string"]
    )
    blob_client = blob_service_client.get_blob_client(
        container=config["azure"]["container_name"], blob=blob_path
    )
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data)


def upload_files_to_blob(file_list, blob_list):
    for ix in range(len(file_list)):
        upload_file_to_blob(file_list[ix], blob_list[ix])


def read_sample_data(
    data_directory,
    file_name,
    n_rows,
    select_columns,
    date_configs,
    file_type,
    file_compression_type=None,
):
    if file_type == "csv":
        df = pd.read_csv(
            os.path.join(data_directory, file_name),
            nrows=n_rows,
            usecols=select_columns,
            compression=file_compression_type,
        )
        for date_col, date_format in date_configs.items():
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        return df


def optimize_numeric_data_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "int64":
            if (
                df[col].min() > np.iinfo(np.int8).min
                and df[col].max() < np.iinfo(np.int8).max
            ):
                df[col] = df[col].astype(np.int8)
            elif (
                df[col].min() > np.iinfo(np.int16).min
                and df[col].max() < np.iinfo(np.int16).max
            ):
                df[col] = df[col].astype(np.int16)
            elif (
                df[col].min() > np.iinfo(np.int32).min
                and df[col].max() < np.iinfo(np.int32).max
            ):
                df[col] = df[col].astype(np.int32)
            elif (
                df[col].min() > np.iinfo(np.int64).min
                and df[col].max() < np.iinfo(np.int64).max
            ):
                df[col] = df[col].astype(np.int64)
        elif df[col].dtype == "float64":
            if (
                df[col].min() > np.finfo(np.float16).min
                and df[col].max() < np.finfo(np.float16).max
            ):
                df[col] = df[col].astype(np.float16)
            elif (
                df[col].min() > np.finfo(np.float32).min
                and df[col].max() < np.finfo(np.float32).max
            ):
                df[col] = df[col].astype(np.float32)
            elif (
                df[col].min() > np.finfo(np.float64).min
                and df[col].max() < np.finfo(np.float64).max
            ):
                df[col] = df[col].astype(np.float64)
    return df


def optimize_categorical_data_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            if df[col].nunique() < 0.5 * len(df):
                df[col] = df[col].astype("category")
    return df


def optimize_df_memory(df: pd.DataFrame) -> pd.DataFrame:
    df = optimize_numeric_data_types(df)
    df = optimize_categorical_data_types(df)
    return df


def generate_optimized_data_schema(data, io_config, parsing_config):
    expected_date_columns = list(parsing_config["date_format_configs"].keys())
    for date_col in expected_date_columns:
        data[date_col] = pd.to_datetime(
            data[date_col], format=parsing_config["date_format_configs"][date_col]
        )
    actual_numeric_columns = data.select_dtypes(
        include=["int64", "float64", "int32", "float32", "int16", "float16", "int8"]
    ).columns.tolist()
    actual_categorical_columns = data.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    expected_numeric_columns = io_config["data_type_dict"]["number"]
    excepted_categorical_columns = io_config["data_type_dict"]["category"]
    missing_numeric_columns = list(
        set(expected_numeric_columns) - set(actual_numeric_columns)
    )
    extra_numeric_columns = list(
        set(actual_numeric_columns) - set(expected_numeric_columns)
    )
    missing_categorical_columns = list(
        set(excepted_categorical_columns) - set(actual_categorical_columns)
    )
    extra_categorical_columns = list(
        set(actual_categorical_columns) - set(excepted_categorical_columns)
    )
    if len(missing_numeric_columns) > 0:
        logger.info("Missing numeric columns: ", missing_numeric_columns)
    if len(extra_numeric_columns) > 0:
        print("Extra numeric columns: ", extra_numeric_columns)
    if len(missing_categorical_columns) > 0:
        print("Missing categorical columns: ", missing_categorical_columns)
    if len(extra_categorical_columns) > 0:
        print("Extra categorical columns: ", extra_categorical_columns)
    data.loc[:, expected_numeric_columns] = optimize_numeric_data_types(
        data[expected_numeric_columns]
    )
    data.loc[:, excepted_categorical_columns] = optimize_categorical_data_types(
        data[excepted_categorical_columns]
    )
    float_columns = data.select_dtypes(include=[float]).columns.tolist()
    data[float_columns] = data[float_columns].round(
        parsing_config["output_data_precision"]
    )
    return data


def read_full_data(
    data_directory,
    file_name,
    select_columns,
    date_configs,
    optimized_schema,
    file_type,
    file_compression_type,
):
    if file_type == "csv":
        df = pd.read_csv(
            os.path.join(data_directory, file_name),
            usecols=select_columns,
            compression=file_compression_type,
            dtype=optimized_schema,
        )
        for date_col, date_format in date_configs.items():
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        return df


def check_cols(df, expected_cols):
    missing_cols = list(set(expected_cols) - set(df.columns))
    extra_cols = list(set(df.columns) - set(expected_cols))
    if len(missing_cols) > 0:
        print("Missing columns: ", missing_cols)
    if len(extra_cols) > 0:
        print("Extra columns: ", extra_cols)


def check_range(data, range_check):
    for col, range in range_check.items():
        try:
            if data[col].min() < range[0] or data[col].max() > range[1]:
                print("Range check failed for column: ", col)
        except:
            # FIXME: Handle data object comparison
            print("Error occurred during range check for: ", col)


def check_uniques(data, unique_check):
    for col, unique_values in unique_check.items():
        if not data[col].isin(unique_values).all():
            print("Unique value check failed for column: ", col)


def check_nulls(data, null_check):
    for col in null_check:
        if data[col].isnull().values.any():
            print("Null check failed for column: ", col)
        else:
            print("Null check passed for column: ", col)


def check_duplicates(data, duplicate_check):
    initial_rows = len(data)
    # This can be problem in distributed environment. check
    unique_rows = data[duplicate_check].drop_duplicates().shape[0]
    if initial_rows != unique_rows:
        print("Duplicate check failed for columns: ", duplicate_check)
    else:
        print("Duplicate check passed for columns: ", duplicate_check)


def read_config(config_directory, file_name, config_type):
    if config_type == "json":
        with open(os.path.join(config_directory, file_name)) as json_file:
            config = json.load(json_file)
        return config


def write_config(config_directory, file_name, config_type, config):
    if config_type == "json":
        with open(os.path.join(config_directory, file_name), "w") as outfile:
            json.dump(config, outfile)


def check_model_id_in_data(data, model_id):
    if model_id in data.columns:
        print("Model ID found in data")
        return True
    else:
        print("Model ID not found in data")
        return False


def check_model_id_constructor_in_data(data, model_id_constructor):
    if set(data.columns).issuperset(set(model_id_constructor)):
        print("Model ID constructor found in data")
        return True
    else:
        print("Model ID constructor not found in data")
        return False


def check_split_character_in_model_id_constructor(
    data, model_id_constructor, split_character
):
    for col in model_id_constructor:
        if split_character in data[col].values:
            print("Split character found in model id constructor")
            return True
    print("Split character not found in model id constructor")
    return False


def extract_data_hash(data, hash_name):
    hash_id = joblib.hash(data, hash_name)
    print("Hash ID: ", hash_id)
    return hash_id


def create_model_id(data, model_id, model_id_constructor, split_character):
    # FIXME: NOTE, this function will be very slow. Ideally this has to be created by the feature engineering team.
    print("This is a slow function will take a lot of time.")
    print("Creating model id")
    data[model_id] = (
        data[model_id_constructor]
        .astype(str)
        .apply(lambda x: split_character.join(x), axis=1)
    )
    data = data.drop(model_id_constructor, axis=1)
    return data


def extract_unique_values(data, hierarchal_columns):
    unique_values = {}
    for col in hierarchal_columns:
        if col in data.select_dtypes(include=["datetime"]).columns.tolist():
            unique_values[col] = data[col].dt.strftime("%Y-%m-%d").unique().tolist()
        else:
            unique_values[col] = data[col].unique().tolist()
    return unique_values


def extract_date_range(data, date_cols):
    date_range = {}
    for col in date_cols:
        date_range[col] = {
            "min_date": data[col].min().strftime("%Y-%m-%d"),
            "max_date": data[col].max().strftime("%Y-%m-%d"),
        }
    return date_range


def extract_range(data, range_cols):
    range = {}
    for col in range_cols:
        range[col] = {"min_value": data[col].min(), "max_value": data[col].max()}
    return range


def extract_unique_counts(data, unique_count_cols):
    unique_counts = {}
    for col in unique_count_cols:
        unique_counts[col] = data[col].nunique()
    return unique_counts


if __name__ == "__main__":

    # Configs
    io_config = {
        "data_directory": "data",
        "config_directory": "configs",
        "model_directory": "models",
        "log_directory": "logs",
        "compression": None,
        "data_format": "csv",
        "select_columns": ["poc_id", "sku_id", "quantity", "date"],
        "data_type_dict": {
            "number": ["quantity"],
            "category": ["poc_id", "sku_id"],
            "date": ["date"],
        },
    }
    parsing_config = {
        "date_format_configs": {"date": "%Y-%m-%d"},
        "input_data_precision": 2,
        "output_data_precision": 2,
        "optimized_data_schema_file_name": "optimized_data_schema.json",
    }
    validator_config = {
        "column_check": ["poc_id", "sku_id", "quantity", "date"],
        "range_check": {"date": ["2020-01-01", "2020-01-05"], "quantity": [0, 2]},
        "unique_check": {
            "poc_id": [
                45,
                66,
                48,
                32,
                80,
                15,
                21,
                1,
                59,
                39,
                73,
                78,
                86,
                2,
                33,
                6,
                41,
                13,
                44,
                43,
                9,
                19,
                95,
                8,
                24,
                54,
                37,
                36,
                96,
                51,
                5,
                52,
                16,
                94,
                65,
                7,
                46,
                72,
                34,
                26,
                88,
                25,
                35,
                11,
                49,
                57,
                55,
                30,
                81,
                89,
                82,
                12,
                17,
                83,
                4,
                76,
                67,
                38,
                93,
                40,
                77,
                53,
                99,
                28,
                60,
                56,
                92,
                47,
                10,
                23,
                63,
                42,
                58,
                62,
                75,
                29,
                31,
                74,
                50,
                68,
                90,
                64,
                61,
                22,
                18,
                70,
                91,
                79,
                20,
                69,
                97,
                14,
                98,
                3,
                84,
                87,
                71,
                27,
                85,
            ],
            "sku_id": [
                48,
                40,
                65,
                75,
                5,
                54,
                59,
                51,
                24,
                42,
                20,
                76,
                32,
                96,
                12,
                10,
                37,
                45,
                84,
                87,
                15,
                80,
                35,
                60,
                81,
                25,
                52,
                69,
                70,
                1,
                90,
                46,
                78,
                47,
                67,
                19,
                50,
                53,
                27,
                7,
                44,
                41,
                94,
                8,
                63,
                92,
                82,
                6,
                61,
                22,
                99,
                89,
                2,
                43,
                18,
                28,
                17,
                31,
                93,
                91,
                34,
                62,
                33,
                23,
                66,
                30,
                14,
                64,
                26,
                36,
                83,
                71,
                97,
                72,
                88,
                3,
                73,
                57,
                9,
                79,
                29,
                11,
                39,
                77,
                98,
                95,
                13,
                55,
                86,
                16,
                38,
                58,
                74,
                21,
                85,
                56,
                68,
                4,
                49,
            ],
        },
        "null_check": ["poc_id", "sku_id", "date"],
        "duplicate_check": ["poc_id", "sku_id", "date"],
    }
    processor_config = {
        "pre_processor": {
            "model_id": "model_id",
            "model_id_constructor": ["poc_id", "sku_id"],
            "model_split_character": "|||",
            "date_freq_configs": {"date": "MS"},
            "aggregation_configs": {},
            "column_mapper": {
                "a": "a",
                "b": "b",
                "c": "c",
                "d": "d",
                "e": "e",
                "f": "f",
                "g": "g",
                "h": "h",
                "i": "i",
            },
        },
        "post_processor": {"reverse_column_mapper": {}},
    }
    metadata_config = {"hash_name": "md5"}
    processor_config["post_processor"]["reverse_column_mapper"] = {
        v: k for k, v in processor_config["pre_processor"]["column_mapper"].items()
    }

    # Data IO
    # Logging
    logger.add(
        os.path.join(io_config["log_directory"], "data_preparation.log"),
        rotation="10 MB",
        level="INFO",
    )
    # Create folders
    folder_list = [
        "data",
        "data/raw",
        "data/interim",
        "data/processed",
        "data/cached",
        "configs",
        "logs",
    ]
    create_folders(folder_list)
    files_list = [
        "data/raw/raw_data.csv",
        "data/raw/test.csv",
        "data/raw/sample_submission.csv",
    ]
    print(check_files_exists(files_list))
    # Download from data store or config store and store it in right place:
    # either read from local FS, data store/blob/db, config store/blob/db, dvc. fs. specific file location in local FS
    df = synthetic_data()
    df.to_csv(
        os.path.join(io_config["data_directory"], "raw", "raw_data.csv"), index=False
    )
    print(df.head())
    show_config()
    sample_data = read_sample_data(
        os.path.join(io_config["data_directory"], "raw"),
        "raw_data.csv",
        100,
        io_config["select_columns"],
        parsing_config["date_format_configs"],
        io_config["data_format"],
        io_config["compression"],
    )
    print(sample_data.head())
    # FIXME: Next line is not working as expected. Why int64 is not getting converted to int32? Debug it.
    type_optimized_data = generate_optimized_data_schema(
        sample_data, io_config, parsing_config
    )
    print(type_optimized_data.head())
    optimized_data_schema = (
        type_optimized_data.select_dtypes(
            include=["number", "category", "bool", "object"]
        )
        .dtypes.apply(lambda x: x.name)
        .to_dict()
    )
    print(optimized_data_schema)
    write_config(
        io_config["config_directory"],
        parsing_config["optimized_data_schema_file_name"],
        optimized_data_schema,
        "json",
    )
    full_data = read_full_data(
        os.path.join(io_config["data_directory"], "raw"),
        "raw_data.csv",
        io_config["select_columns"],
        parsing_config["date_format_configs"],
        optimized_data_schema,
        io_config["data_format"],
        io_config["compression"],
    )
    print(full_data.head())
    check_cols(full_data, validator_config["column_check"])
    check_range(full_data, validator_config["range_check"])
    # FIXME: This should be already done during data parsing. Why is it not working?
    full_data["poc_id"] = full_data["poc_id"].astype("category")
    full_data["sku_id"] = full_data["sku_id"].astype("category")
    check_uniques(full_data, validator_config["unique_check"])
    check_nulls(full_data, validator_config["null_check"])
    check_duplicates(full_data, validator_config["duplicate_check"])
    print(extract_data_hash(full_data, metadata_config["hash_name"]))
    print(
        extract_date_range(
            full_data, list(parsing_config["date_format_configs"].keys())
        )
    )
    print(
        extract_range(
            full_data, full_data.select_dtypes(include=["number"]).columns.tolist()
        )
    )
    print(
        extract_unique_values(
            full_data,
            list(validator_config["unique_check"].keys())
            + list(parsing_config["date_format_configs"].keys()),
        )
    )
    print(
        extract_unique_counts(
            full_data,
            list(validator_config["unique_check"].keys())
            + list(parsing_config["date_format_configs"].keys()),
        )
    )
    if not check_model_id_in_data(
        full_data, processor_config["pre_processor"]["model_id"]
    ):
        if check_model_id_constructor_in_data(
            full_data, processor_config["pre_processor"]["model_id_constructor"]
        ):
            if not check_split_character_in_model_id_constructor(
                full_data,
                processor_config["pre_processor"]["model_id_constructor"],
                processor_config["pre_processor"]["model_split_character"],
            ):
                full_data = create_model_id(
                    full_data,
                    processor_config["pre_processor"]["model_id"],
                    processor_config["pre_processor"]["model_id_constructor"],
                    processor_config["pre_processor"]["model_split_character"],
                )
