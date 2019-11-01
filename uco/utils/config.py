import yaml


def load_config(filename: str) -> dict:
    """
    Load a configuration file as YAML and assign the experiment a verbose name.
    """
    with open(filename) as fh:
        config = yaml.safe_load(fh)
    return config


def load_train_config(filename: str) -> dict:
    """
    Load a configuration file as YAML and assign the experiment a verbose name.
    """
    config = load_config(filename)
    config["name"] = verbose_config_name(config)
    return config


def verbose_config_name(config: dict) -> str:
    """
    Construct a verbose name for an experiment by extracting configuration settings.
    """
    short_name = config["short_name"]
    arch = f"{config['arch']['type']}-{config['arch']['args']['encoder_name']}"
    loss = config["loss"]["type"]
    return "-".join([short_name, arch, loss])
