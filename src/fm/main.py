import pathlib

from typer import Typer
from fm.preprocess import preprocess
from fm.train import train
from fm.logger import make_logger

LOCAL_DIRECTORY_PATH = "/tmp/fm"

local_directory = pathlib.Path(LOCAL_DIRECTORY_PATH)
logger = make_logger()
app = Typer()


@app.command("preprocess")
def run_preprocess():
    preprocess(local_directory, logger)


@app.command("train")
def run_train():
    train(local_directory, logger)


if __name__ == "__main__":
    app()
