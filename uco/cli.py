import click

from uco.main import EnsembleManager, Runner
from uco.ensemble import HDF5PredictionReducer
from uco.utils import load_config, load_train_config, kaggle_submit, setup_logging


@click.group()
def cli():
    """
    CLI for uco
    """


@cli.command()
@click.option('-i', '--inference-config-filename', default='experiments/inference.yml',
              help='Path to inference configuration')
@click.option('-n', '--num-models', default=5, help='How many models should be in the ensemble?')
def train_ensemble(inference_config_filename, num_models):
    config = load_config(inference_config_filename)
    EnsembleManager(config).start(num_models)


@cli.command()
@click.option('-p', '--predictions-filename', type=str,
              default='data/predictions/raw-predictions.h5')
@click.option('-d', '--data-directory', type=str, default='data/raw')
def average(predictions_filename, data_directory):
    HDF5PredictionReducer().average(predictions_filename, data_directory, 'submission.csv')


@cli.command()
@click.option('-f', '--filename', type=str,
              default='data/predictions/submission.csv')
@click.option('-n', '--name', type=str, required=True)
def submit(filename, name):
    kaggle_submit(filename, name)


@cli.command()
@click.option('-c', '--config-filename', default=['experiments/config.yml'], multiple=True,
              help=('Path to training configuration file. If multiple are provided, runs will be '
                    'executed in order'))
@click.option('-r', '--resume', default=None, type=str,
              help='path to latest checkpoint (default: None)')
def train(config_filename, resume):
    """
    Entry point to start training run(s).
    """
    configs = [load_train_config(f) for f in config_filename]
    for config in configs:
        Runner(config).train(resume)


@cli.command()
@click.option('-c', '--config-filename', default='experiments/inference.yml',
              help='Path to training configuration file.')
@click.option('-m', '--model-checkpoint', required=True, type=str,
              help='Model checkpoint to run inference using.')
def predict(config_filename, model_checkpoint):
    """
    Perform inference using saved model weights, and save to HDF5 database.
    """
    config = load_config(config_filename)
    Runner(config).predict(model_checkpoint)
