import click
import yaml

from uco.main import Runner


@click.group()
def cli():
    """
    CLI for uco
    """


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
    configs = [load_config(f) for f in config_filename]
    for config in configs:
        Runner(config).train(resume)


def load_config(filename: str) -> dict:
    """
    Load a configuration file as YAML and assign the experiment a verbose name.
    """
    with open(filename) as fh:
        config = yaml.safe_load(fh)
    config['name'] = verbose_config_name(config)
    return config


def verbose_config_name(config: dict) -> str:
    """
    Construct a verbose name for an experiment by extracting configuration settings.
    """
    short_name = config['short_name']
    arch = f"{config['arch']['type']}-{config['arch']['args']['encoder_name']}"
    loss = config['loss']['type']
    optim = config['optimizer']['type']
    return '-'.join([short_name, arch, loss, optim])
