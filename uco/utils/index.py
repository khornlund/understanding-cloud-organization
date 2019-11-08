from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm


class Indexer:

    checkpoint_partial_path = "checkpoints/model_best.pth"
    index_filename = "index.csv"

    @classmethod
    def reindex(cls, path):
        path = Path(path)

        items = []
        for child in tqdm(path.iterdir()):
            if not child.is_dir():
                continue
            try:
                items.append(cls.extract_config(child))
            except Exception as ex:
                print(f"{child}: {ex}")
        df = pd.DataFrame.from_records(items)
        df.sort_values("run", inplace=True)
        save_as = path / cls.index_filename
        df.to_csv(save_as, index=False)

    @classmethod
    def index(cls, run):
        config = cls.extract_config(run)
        parent = run.parent
        index_filename = parent / cls.index_filename
        try:
            df = pd.read_csv(index_filename)
            df = df.append(config, ignore_index=True)
        except FileNotFoundError:
            df = pd.DataFrame.from_records([config])
        df.sort_values("run", inplace=True)
        df.to_csv(index_filename, index=False)

    @classmethod
    def extract_config(cls, run_dir):
        checkpoint = torch.load(
            run_dir / cls.checkpoint_partial_path, map_location=torch.device("cpu")
        )

        best_score = checkpoint["monitor_best"].item()
        train_cfg = checkpoint["config"]

        settings = {
            "mean_dice": best_score,
            "run": run_dir.name,
            "encoder": train_cfg["arch"]["args"]["encoder_name"],
            "decoder": train_cfg["arch"]["type"],
            "dropout": train_cfg["arch"]["args"].get("dropout", 0),
            "augs": train_cfg["augmentation"]["type"],
            "img_height": train_cfg["augmentation"]["args"]["height"],
            "img_width": train_cfg["augmentation"]["args"]["width"],
            "batch_size": train_cfg["data_loader"]["args"]["batch_size"],
            "bce_weight": train_cfg["loss"]["args"]["bce_weight"],
            "dice_weight": train_cfg["loss"]["args"].get("dice_weight", 0),
            "lovasz_weight": train_cfg["loss"]["args"].get("lovasz_weight", 0),
            "optimizer": train_cfg["optimizer"]["type"],
            "encoder_lr": train_cfg["optimizer"]["encoder"]["lr"],
            "decoder_lr": train_cfg["optimizer"]["decoder"]["lr"],
            "anneal_start": train_cfg["lr_scheduler"]["args"]["start_anneal"],
            "anneal_end": train_cfg["lr_scheduler"]["args"]["n_epochs"],
            "encoder_weights": train_cfg["arch"]["args"]["encoder_weights"],
            "seed": train_cfg["seed"],
        }
        return settings
