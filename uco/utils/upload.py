import subprocess
from pathlib import Path

from .config import load_config


COMPETITION_NAME = 'understanding_cloud_organization'


def kaggle_submit(csv_filename):
    # create the directory to contain our new dataset
    csv_filename = Path(csv_filename)
    config_filename = csv_filename.parent / 'checkpoints/config.yaml'

    # use the name generated from config as submit msg
    config = load_config(config_filename)
    msg = config['name']

    # upload
    p = subprocess.Popen(
        [
            'kaggle', 'competitions', 'submit',
            '-f', str(csv_filename),
            '-m', msg,
            COMPETITION_NAME
        ],
    )
    p.communicate()
