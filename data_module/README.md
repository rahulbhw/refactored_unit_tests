```python
strategy_config = {
    "service_name": "suggested order",
    "strategy_name": "holt winters",
    "strategy_type": "strategy execution",
    "country": "dr",
    "config_version": "1.0",
    "model_id": "model_id",
}
```

* `service_name`: Name of the service (`so`, `up`, etc.)
* `strategy_name`: Name of the strategy (`holt winters`, etc.)
* `strategy_type`: Type of the strategy  (`strategy execution`, `backtesting`, `HPO`, `AutoML` etc.)
* `country`: Country of the service (`dr`, `mx`, etc.)
* `config_version`: Version of the config
* `model_id`: hash of the model object + pipeline if existing or `None`.


```python
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
        "boolean": None,
    },
}
```

* `data_directory`: Directory where the data is stored
* `config_directory`: Directory where the config files are stored
* `model_directory`: Directory where the model files are stored
* `log_directory`: Directory where the log files are stored
* `compression`: Compression of the data (`None`, `gzip`, `bz2`, etc.)
* `data_format`: Format of the data (`csv`, `parquet`, etc.)
* `select_columns`: Columns to be selected from the data
* `data_type_dict`: Dictionary of data types (`number`, `category`, `date`, `boolean`)

__Note:__ Keys of `data_type_dict` should be the same as `select_columns`. If not stop execution and raise an error. `data_type_dict` should be parsable otherwise stop execution and raise an error. if `data_directory`, `config_directory`, `model_directory`, `log_directory` are not present then create them. Also, we need to have a list of files which has to be present in local to run the pipeline. If not present then show an warning stating that the file is not present and the pipeline will not run if the values expected from file path has not been supplied during runtime execution. If required files are not present in file path and not supplied during runtime execution then stop execution and raise an error. 


```python
parsing_config = {
    "date_format_configs": {"date": "%Y-%m-%d"},
    "input_data_precision": 2,
    "output_data_precision": 2,
    "optimized_data_schema_file_name": "optimized_data_schema.json",
}
```

* `date_format_configs`: Dictionary of date formats (date, datetime, etc.)
* `input_data_precision`: Precision of the input data. Use this after parsing the data to numbers for float values.
* `output_data_precision`: Precision of the output data. Use this before writing the data to files for float values.
* `optimized_data_schema_file_name`: Name of the optimized data schema file


__Note:__ keys of `date_format_configs` should match with `list(io_config['data_type_dict'].keys())`. If not stop execution and raise an error. `date_format_configs` should be parsable otherwise stop execution and raise an error. Any user input while modelling with respect to date has to be validated against `date_format_configs` for a given column. Also, use the same info while saving the data as csv file. `list(io_config['data_type_dict'].values())` has to be validated with valid date formats. If not stop execution and raise an error.

```python
validator_config = {
    "column_check": ["poc_id", "sku_id", "quantity", "date"],
    "range_check": {"date": ["2020-01-01", "2020-01-05"], "quantity": [0, 2]},
    "unique_check": {
        "poc_id": [
            45,
            66,
            48,
        ],
        "sku_id": [
            48,
            40,
            65,
            75,
            ],
    },
    "null_check": ["poc_id", "sku_id", "date"],
    "duplicate_check": ["poc_id", "sku_id", "date"],
}
```

* `column_check`: List of columns to be checked for existence. This is a subset of `select_columns` from `io_config`
* `range_check`: Dictionary of columns to be checked for range. The values are a list of min and max values. A validation should be there to check if min is less than max. If not stop execution and raise an error. also, in case of date columns, date format should be consistent. If not stop execution and raise an error. If None, then skip the check. keys should be a subset of `column_check`
* `unique_check`: Dictionary of columns to be checked for uniqueness. The values are a list of unique values. If not stop execution and raise an error. If None, then skip the check. keys should be a subset of `column_check`
* `null_check`: List of columns to be checked for null values. If None skip. If false, then show an warning. should be a subset of `column_check`
* `duplicate_check`: List of columns to be checked for duplicate values. If None skip. If false, then show an warning. should be a subset of `column_check`

```python
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
```

* `pre_processor`: Dictionary of pre processor configs
    * `model_id`: Name of the column which will be used as `model_id`. If value, then check value is present in data if not stop execution and raise an error. If None, then create a new column with name `model_id`
    * `model_id_constructor`: List of columns which will be used to construct `model_id`. If `model_id` is not `None`, and values are present in `model_id_constructor` then then show an warning what model_id will be reconstructed using this and this is a slow operation. We expect that this will be created at feature store level and will be present in the data. 
    * `model_split_character`: Character which will be used to split the model id. Validate this is not present in any of the model_id_constructor columns. If not stop execution and raise an error. If not present then use `|||` to create `model_id`. Use the same for reverse mapping.
    * `date_freq_configs`: Dictionary of date frequencies which has to be applied if the date_frequency to be changed while pre-processing. Also, we need to take input that if this frequency changes the how to broadcast the data or aggregate.
    * `aggregation_configs`: __WIP__
    * `column_mapper`: __WIP__
* `post_processor`: __WIP__
    * `reverse_column_mapper`: __WIP__