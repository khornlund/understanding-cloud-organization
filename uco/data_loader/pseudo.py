from pathlib import Path

import pandas as pd

from .process import RLEConverter
from .data_loaders import pivot_df


class PseudoLabelBuilder:
    @classmethod
    def build(cls, submission_filename, data_dir):
        df = pd.read_csv(submission_filename)

        df_pivot = pivot_df(df)
        empty_imgs = df_pivot.loc[df_pivot.n_classes == 0, :].index
        to_drop = []
        for filename in empty_imgs:
            for label in ["Fish", "Flower", "Gravel", "Sugar"]:
                to_drop.append(f"{filename}_{label}")

        df.set_index("Image_Label", inplace=True)
        df.drop(to_drop, axis=0)

        print(f"Upscaling labels")
        df["EncodedPixels"] = df["EncodedPixels"].apply(RLEConverter.upscale_single)
        save_as = Path(data_dir) / "pseudo.csv"
        print(f"Saving {df.shape} to {save_as}")
        df.to_csv(save_as)
