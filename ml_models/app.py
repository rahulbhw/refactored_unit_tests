import typer
from typing import Optional
from watermark import watermark
from utils import get_system_info

app = typer.Typer()


@app.command()
def get_watermark():
    typer.echo(
        watermark(
            packages="numpy,scipy,sklearn,pandas,joblib,ray,dask,ipyparallel,joblibspark,pyspark,tune_sklearn"
        )
    )
    typer.echo(get_system_info())


@app.command()
def example_cmd_args(
    name: Optional[str] = typer.Argument(None),
    last_name: Optional[str] = typer.Argument(None),
    age: Optional[str] = typer.Argument(None),
):
    print(f"name:{name}  last-name:{last_name}  age:{age}")


def main():
    app()


if __name__ == "__main__":
    main()
