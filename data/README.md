```bash
conda create -n aikit-modin python=3.8 intel-aikit-modin -c intel -c conda-forge -y
conda activate aikit-modin
conda install scikit-learn-intelex -c conda-forge
pip install ipyparallel joblib jupyter notebook scikit-learn pandas numpy "dask[distributed]" watermark joblibspark pyspark tune-sklearn typer "modin[all]"
pip list --format=freeze > requirements_intel.txt
conda deactivate
```