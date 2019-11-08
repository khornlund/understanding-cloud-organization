import click
from pathlib import Path

from uco.ensemble import EnsembleManager
from uco.runner import TrainingManager, InferenceManager
from uco.h5 import HDF5AverageWriter, PostProcessor
from uco.utils import (
    load_config,
    load_train_config,
    kaggle_submit,
    Indexer,
    setup_logging,
)


@click.group()
def cli():
    """
    CLI for uco
    """


@cli.command()
@click.option(
    "-c",
    "--config-filename",
    default=["experiments/config.yml"],
    multiple=True,
    help=(
        "Path to training configuration file. If multiple are provided, runs will be "
        "executed in order"
    ),
)
@click.option(
    "-r",
    "--resume",
    default=None,
    type=str,
    help="path to latest checkpoint (default: None)",
)
def train(config_filename, resume):
    """
    Entry point to start training run(s).
    """
    configs = [load_train_config(f) for f in config_filename]
    for config in configs:
        TrainingManager(config).run(resume)


@cli.command()
@click.option(
    "-i",
    "--inference-config-filename",
    default="experiments/inference.yml",
    help="Path to inference configuration",
)
@click.option(
    "-n", "--num-models", default=5, help="How many models should be in the ensemble?"
)
def train_ensemble(inference_config_filename, num_models):
    config = load_config(inference_config_filename)
    EnsembleManager(config).start(num_models)


@cli.command()
@click.option("-f", "--folder", type=str, required=True, help="Folder to index")
def reindex(folder):
    Indexer.reindex(folder)


@cli.command()
@click.option(
    "-c",
    "--config-filename",
    default="experiments/inference.yml",
    help="Path to training configuration file.",
)
@click.option(
    "-m",
    "--model-checkpoint",
    required=True,
    type=str,
    help="Model checkpoint to run inference using.",
)
def predict(config_filename, model_checkpoint):
    """
    Perform inference using saved model weights, and save to HDF5 database.
    """
    config = load_config(config_filename)
    InferenceManager(config).run(model_checkpoint)


@cli.command()
@click.option(
    "-f",
    "--folder",
    type=str,
    default="saved/training",
    help="Folder containing checkpoints",
)
def predict_all(folder):
    config = load_config("experiments/inference.yml")
    checkpoints = sorted(list(Path(folder).glob("**/*model_best.pth")))
    print(f"Performing predictions for {checkpoints}")
    for checkpoint in checkpoints:
        InferenceManager(config).run(checkpoint)


@cli.command()
@click.option(
    "-c",
    "--config-filename",
    default="experiments/inference.yml",
    help="Path to training configuration file.",
)
def average(config_filename):
    config = load_config(config_filename)
    setup_logging(config)
    HDF5AverageWriter(verbose=config["verbose"]).average(
        config["output"]["raw"], config["output"]["avg"]
    )


@cli.command()
@click.option(
    "-c",
    "--config-filename",
    default="experiments/inference.yml",
    help="Path to training configuration file.",
)
def post_process(config_filename):
    config = load_config(config_filename)
    setup_logging(config)
    PostProcessor(verbose=config["verbose"]).process(
        config["output"]["avg"],
        config["data_loader"]["args"]["data_dir"],
        config["output"]["sub"],
    )


@cli.command()
@click.option("-f", "--filename", type=str, default="data/predictions/submission.csv")
@click.option("-n", "--name", type=str, required=True)
def submit(filename, name):
    kaggle_submit(filename, name)
