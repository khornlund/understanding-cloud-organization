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

        best_score = checkpoint["monitor_best"]
        if isinstance(best_score, torch.Tensor):
            best_score = best_score.item()
        cfg = checkpoint["config"]

        settings = {
            "score": best_score,
            "run": run_dir.name,
            "seed": cfg["seed"],
            "encoder": cfg["arch"]["args"].get("encoder_name", ""),
            "decoder": cfg["arch"]["type"],
            "dropout": cfg["arch"]["args"].get("dropout", 0),
            "encoder_weights": cfg["arch"]["args"].get("encoder_weights", ""),
            "augs": cfg["augmentation"]["type"],
            "img_height": cfg["augmentation"]["args"]["height"],
            "img_width": cfg["augmentation"]["args"]["width"],
            "batch_size": cfg["data_loader"]["args"]["batch_size"],
            "bce_weight": cfg["loss"]["args"].get("bce_weight", 1),
            "dice_weight": cfg["loss"]["args"].get("dice_weight", 0),
            "lovasz_weight": cfg["loss"]["args"].get("lovasz_weight", 0),
            "optimizer": cfg["optimizer"]["type"],
            "anneal_start": cfg["lr_scheduler"]["args"]["start_anneal"],
            "anneal_end": cfg["lr_scheduler"]["args"]["n_epochs"],
        }
        if "encoder" in cfg["optimizer"].keys():
            settings.update(
                {
                    "encoder_lr": cfg["optimizer"]["encoder"]["lr"],
                    "decoder_lr": cfg["optimizer"]["decoder"]["lr"],
                }
            )
        else:
            settings.update(
                {"encoder_lr": cfg["optimizer"]["args"]["lr"], "decoder_lr": 0}
            )
        return settings
