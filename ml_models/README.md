# __This is main__

```bash
conda create -n hpo_bechmark_2023 python=3.8 -y
conda activate hpo_bechmark_2023
pip install ipyparallel joblib jupyter notebook scikit-learn pandas numpy 'ray[all]' "dask[distributed]" watermark joblibspark pyspark tune-sklearn typer loguru
pip list --format=freeze > requirements.txt
conda deactivate
```

```bash
conda create -n rapids-23.02 -c rapidsai -c conda-forge -c nvidia cudf=23.02 cuml=23.02 cugraph=23.02 cuspatial=23.02 cuxfilter=23.02 cusignal=23.02 cucim=23.02 python=3.8 cudatoolkit=11.8
conda activate rapids-23.02
pip install ipyparallel joblib jupyter notebook scikit-learn pandas numpy 'ray[all]' "dask[distributed]" watermark joblibspark pyspark tune-sklearn typer loguru
pip list --format=freeze > requirements_gpu.txt
conda deactivate
```

```bash
conda create -n aikit-modin python=3.8 intel-aikit-modin -c intel -c conda-forge -y
conda activate aikit-modin
conda install scikit-learn-intelex -c conda-forge
pip install ipyparallel joblib jupyter notebook scikit-learn pandas numpy "dask[distributed]" watermark joblibspark pyspark tune-sklearn typer "modin[all]" loguru
pip list --format=freeze > requirements_intel.txt
conda deactivate
```


```bash
.
├── 01_env_setup.mp4
├── 02_requirements_creation.mp4
├── 03_system_library_info.mp4
├── app.py
├── config.py
├── data_processing.py
├── hpo_gpu.py
├── hpo.py
├── __init__.py
├── README.md
├── regression_data.csv.gzip
├── requirements.txt
└── utils.py
```
<!-- https://thenewstack.io/intel-oneapis-unified-programming-model-for-python-machine-learning/ -->
<!-- https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-of-modin.html?utm_source=thenewstack&utm_medium=website&utm_content=inline-mention&utm_campaign=platform#gs.r2c4tc -->