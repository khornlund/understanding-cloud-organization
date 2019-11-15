import shutil

import click
import torch
from pathlib import Path
from tqdm import tqdm

from uco.data_loader import PseudoLabelBuilder
from uco.ensemble import EnsembleManager
from uco.runner import TrainingManager, InferenceManager
from uco import h5
from uco.download import MapDownloader
from uco.utils import (
    load_config,
    load_train_config,
    kaggle_submit,
    Indexer,
    setup_logging,
)


DEFAULT_SEG_INFERENCE = "experiments/original/seg/inference.yml"
DEFAULT_CLAS_INFERENCE = "experiments/original/clas/inference.yml"


@click.group()
def cli():
    """
    CLI for uco
    """


@cli.command()
@click.option(
    "-c",
    "--config-filename",
    default=["experiments/original/seg/train.yml"],
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
    "-c",
    "--config-filename",
    default=DEFAULT_SEG_INFERENCE,
    help="Path to inference configuration",
)
@click.option(
    "-n", "--num-models", default=5, help="How many models should be in the ensemble?"
)
def train_ensemble(config_filename, num_models):
    config = load_config(config_filename)
    EnsembleManager(config).start(num_models)


@cli.command()
@click.option("-f", "--folder", type=str, required=True, help="Folder to index")
def reindex(folder):
    Indexer.reindex(folder)


@cli.command()
@click.option(
    "-c",
    "--config-filename",
    default=DEFAULT_SEG_INFERENCE,
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
    default="saved/original/seg/training",
    help="Folder containing checkpoints",
)
@click.option(
    "-c",
    "--config-filename",
    default=DEFAULT_SEG_INFERENCE,
    help="Path to inference configuration file.",
)
def predict_all(folder, config_filename):
    config = load_config(config_filename)
    checkpoints = sorted(list(Path(folder).glob("**/model_best.pth")))
    print(f"Performing predictions for {checkpoints}")
    for checkpoint in checkpoints:
        try:
            InferenceManager(config).run(checkpoint)
        except Exception as ex:
            print(f"Caught exception: {ex}")


@cli.command()
@click.option(
    "-c",
    "--config-filename",
    default=DEFAULT_SEG_INFERENCE,
    help="Path to training configuration file.",
)
def average(config_filename):
    config = load_config(config_filename)
    setup_logging(config)
    getattr(h5, config["average"])(
        config["output"]["N"], verbose=config["verbose"]
    ).average(config["output"]["raw"], config["output"]["avg"])


@cli.command()
@click.option(
    "-s",
    "--seg-config-filename",
    default=DEFAULT_SEG_INFERENCE,
    help="Path to segmentation inference configuration file.",
)
@click.option(
    "-c",
    "--clas-config-filename",
    default=DEFAULT_CLAS_INFERENCE,
    help="Path to classification inference configuration file.",
)
def post_process(seg_config_filename, clas_config_filename):
    seg_config = load_config(seg_config_filename)
    clas_config = load_config(clas_config_filename)
    setup_logging(seg_config)
    h5.PostProcessor(seg_config["output"]["N"], verbose=2).process(
        seg_config["output"]["avg"],
        clas_config["output"]["avg"],
        seg_config["output"]["img"],
        seg_config["output"]["sub"],
    )


@cli.command()
@click.option("-f", "--filename", type=str, default="data/original/submission.csv")
@click.option("-n", "--name", type=str, required=True)
def submit(filename, name):
    kaggle_submit(filename, name)


@cli.command()
@click.option(
    "-f", "--folder", type=str, required=True, help="Folder containing checkpoints"
)
@click.option("-s", "--score-cutoff", default=0.605, help="delete runs with low scores")
def prune_seg(folder, score_cutoff):
    checkpoints = sorted(list(Path(folder).glob("**/*model_best.pth")))[:-2]
    counter = 0
    for c in tqdm(checkpoints):
        try:
            state_dict = torch.load(c, map_location=torch.device("cpu"))
            if state_dict["monitor_best"] < score_cutoff:
                parent = c.parent.parent
                if "training" in parent.name:  # safety
                    raise Exception("About to delete training directory!")
                shutil.rmtree(parent)
                counter += 1
        except Exception as ex:
            print(f"Caught exception for {c}: {ex}")
    print(f"Deleted {counter}/{len(checkpoints)} checkpoints")


@cli.command()
@click.option(
    "-f", "--folder", type=str, required=True, help="Folder containing checkpoints"
)
@click.option("-s", "--score-cutoff", default=0.490, help="delete runs with high loss")
def prune_clas(folder, score_cutoff):
    checkpoints = sorted(list(Path(folder).glob("**/*model_best.pth")))[:-2]
    counter = 0
    for c in tqdm(checkpoints):
        try:
            state_dict = torch.load(c, map_location=torch.device("cpu"))
            if state_dict["monitor_best"] > score_cutoff:
                parent = c.parent.parent
                if "training" in parent.name:  # safety
                    raise Exception("About to delete training directory!")
                shutil.rmtree(parent)
                counter += 1
        except Exception as ex:
            print(f"Caught exception for {c}: {ex}")
    print(f"Deleted {counter}/{len(checkpoints)} checkpoints")


@cli.command()
@click.option(
    "-s", "--submission-filename", type=str, default="data/pseudo/submission.csv"
)
@click.option("-d", "--data-directory", type=str, default="data/raw")
def create_pseudo(submission_filename, data_directory):
    PseudoLabelBuilder.build(submission_filename, data_directory)


@cli.command()
@click.option("-o", "--output-directory", default="data/raw/gibs", type=str)
@click.option("-n", "--num-workers", default=4, type=int)
def download_gibs(output_directory, num_workers):
    MapDownloader(output_directory, num_workers).download()
