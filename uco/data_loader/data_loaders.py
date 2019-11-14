from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from uco.base import DataLoaderBase

from .datasets import (
    CloudDatasetTrainVal,
    CloudDatasetTest,
    CloudClasDatasetTrainVal,
    CloudClasDatasetTest,
    CloudDatasetPseudoTrain,
    CloudDatasetPseudoTest,
)
from .sampler import SamplerFactory


class CloudSegDataLoader(DataLoaderBase):

    train_csv = "train.csv"

    def __init__(
        self,
        transforms,
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        nworkers,
        pin_memory=True,
    ):
        self.transforms, self.shuffle = transforms, shuffle
        self.bs, self.nworkers, self.pin_memory = batch_size, nworkers, pin_memory
        self.data_dir = Path(data_dir)

        self.train_df, self.val_df = self.load_df(validation_split)

        tsfm = self.transforms.build(train=True)
        dataset = CloudDatasetTrainVal(self.train_df, self.data_dir, tsfm)

        super().__init__(
            dataset,
            self.bs,
            shuffle=shuffle,
            num_workers=self.nworkers,
            pin_memory=self.pin_memory,
        )

    def load_df(self, validation_split):
        df = pd.read_csv(self.data_dir / self.train_csv)
        df = pivot_df(df)
        if validation_split > 0:
            return train_test_split(
                df, test_size=validation_split, stratify=df["n_classes"]
            )
        return df, pd.DataFrame({})

    def split_validation(self):
        if self.val_df.empty:
            return None
        else:
            tsfm = self.transforms.build(train=False)
            dataset = CloudDatasetTrainVal(self.val_df, self.data_dir, tsfm)
            return DataLoader(
                dataset,
                self.bs * 2,
                num_workers=self.nworkers,
                pin_memory=self.pin_memory,
            )


class CloudSegTestDataLoader(DataLoaderBase):

    test_csv = "sample_submission.csv"

    def __init__(self, transforms, data_dir, batch_size, nworkers, pin_memory=True):
        self.bs, self.nworkers, self.pin_memory = batch_size, nworkers, pin_memory
        self.data_dir = Path(data_dir)
        self.test_df = self.load_df()
        tsfm = transforms.build(train=False)
        dataset = CloudDatasetTest(self.test_df, self.data_dir, tsfm)
        super().__init__(
            dataset,
            self.bs,
            shuffle=False,
            num_workers=self.nworkers,
            pin_memory=self.pin_memory,
        )

    def load_df(self):
        df = pd.read_csv(self.data_dir / self.test_csv)
        df = df.iloc[np.arange(0, df.shape[0], step=4)]  # filenames are repeated 4x
        df.columns = ["Image", "EncodedPixels"]
        df["Image"] = df["Image"].apply(lambda f: f.split("_")[0])
        df.set_index("Image", inplace=True)
        return df


class CloudClasDataLoader(DataLoaderBase):

    train_csv = "train.csv"

    def __init__(
        self,
        transforms,
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        nworkers,
        pin_memory=True,
    ):
        self.transforms, self.shuffle = transforms, shuffle
        self.bs, self.nworkers, self.pin_memory = batch_size, nworkers, pin_memory
        self.data_dir = Path(data_dir)

        self.train_df, self.val_df = self.load_df(validation_split)

        tsfm = self.transforms.build(train=True)
        dataset = CloudClasDatasetTrainVal(self.train_df, self.data_dir, tsfm)

        super().__init__(
            dataset,
            self.bs,
            shuffle=shuffle,
            num_workers=self.nworkers,
            pin_memory=self.pin_memory,
        )

    def load_df(self, validation_split):
        df = pd.read_csv(self.data_dir / self.train_csv)
        df = pivot_df(df)
        if validation_split > 0:
            return train_test_split(
                df, test_size=validation_split, stratify=df["n_classes"]
            )
        return df, pd.DataFrame({})

    def split_validation(self):
        if self.val_df.empty:
            return None
        else:
            tsfm = self.transforms.build(train=False)
            dataset = CloudClasDatasetTrainVal(self.val_df, self.data_dir, tsfm)
            return DataLoader(
                dataset,
                self.bs * 2,
                num_workers=self.nworkers,
                pin_memory=self.pin_memory,
            )


class CloudClasTestDataLoader(DataLoaderBase):

    test_csv = "sample_submission.csv"

    def __init__(self, transforms, data_dir, batch_size, nworkers, pin_memory=True):
        self.bs, self.nworkers, self.pin_memory = batch_size, nworkers, pin_memory
        self.data_dir = Path(data_dir)
        self.test_df = self.load_df()
        tsfm = transforms.build(train=False)
        dataset = CloudClasDatasetTest(self.test_df, self.data_dir, tsfm)
        super().__init__(
            dataset,
            self.bs,
            shuffle=False,
            num_workers=self.nworkers,
            pin_memory=self.pin_memory,
        )

    def load_df(self):
        df = pd.read_csv(self.data_dir / self.test_csv)
        df = df.iloc[np.arange(0, df.shape[0], step=4)]  # filenames are repeated 4x
        df.columns = ["Image", "EncodedPixels"]
        df["Image"] = df["Image"].apply(lambda f: f.split("_")[0])
        df.set_index("Image", inplace=True)
        return df


def pivot_df(df):
    df["Image"], df["Label"] = zip(*df["Image_Label"].str.split("_"))
    df = df.pivot(index="Image", columns="Label", values="EncodedPixels")
    df.columns = [f"rle{c}" for c in range(4)]
    df["n_classes"] = df.count(axis=1)

    # add classification columns
    for c in range(4):
        df[f"c{c}"] = df[f"rle{c}"].apply(lambda rle: not pd.isnull(rle))
    return df


class CloudSegPseudoDataLoader(DataLoaderBase):

    train_csv = "train.csv"
    pseudo_csv = "pseudo.csv"

    def __init__(
        self,
        transforms,
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        nworkers,
        n_pseudo_per_batch,
        pin_memory=True,
    ):
        self.transforms, self.shuffle = transforms, shuffle
        self.bs, self.nworkers, self.pin_memory = batch_size, nworkers, pin_memory
        self.data_dir = Path(data_dir)

        train_df, self.val_df = self.load_df(True, validation_split)
        pseudo_df, _ = self.load_df(False, 0)
        self.train_df = pd.concat([train_df, pseudo_df])

        tsfm = self.transforms.build(train=True)
        dataset = CloudDatasetPseudoTrain(self.train_df, self.data_dir, tsfm)

        n_train = train_df.shape[0]
        n_pseudo = pseudo_df.shape[0]
        train_idxs = [idx for idx in range(n_train)]
        pseudo_idxs = [idx + n_train for idx in range(n_pseudo)]
        class_idxs = [train_idxs, pseudo_idxs]
        sampler = SamplerFactory(2).get(class_idxs, batch_size, n_pseudo_per_batch)

        super().__init__(
            dataset, batch_sampler=sampler, num_workers=nworkers, pin_memory=pin_memory
        )

    def load_df(self, train, validation_split):
        csv_filename = self.train_csv if train else self.pseudo_csv
        df = pd.read_csv(self.data_dir / csv_filename)
        df = pivot_df(df)
        if train and validation_split > 0:
            return train_test_split(
                df, test_size=validation_split, stratify=df["n_classes"]
            )
        return df, pd.DataFrame({})

    def split_validation(self):
        if self.val_df.empty:
            return None
        else:
            tsfm = self.transforms.build(train=False)
            dataset = CloudDatasetPseudoTrain(self.val_df, self.data_dir, tsfm)
            return DataLoader(
                dataset,
                self.bs * 2,
                num_workers=self.nworkers,
                pin_memory=self.pin_memory,
            )


class CloudSegPseudoTestDataLoader(CloudSegTestDataLoader):

    test_csv = "gibs.csv"

    def __init__(self, transforms, data_dir, batch_size, nworkers, pin_memory=True):
        self.bs, self.nworkers, self.pin_memory = batch_size, nworkers, pin_memory
        self.data_dir = Path(data_dir)
        self.test_df = self.load_df()
        tsfm = transforms.build(train=False)
        dataset = CloudDatasetPseudoTest(self.test_df, self.data_dir, tsfm)
        super().__init__(
            dataset,
            self.bs,
            shuffle=False,
            num_workers=self.nworkers,
            pin_memory=self.pin_memory,
        )

    def load_df(self):
        df = pd.read_csv(self.data_dir / self.test_csv)
        df = df.iloc[np.arange(0, df.shape[0], step=4)]  # filenames are repeated 4x
        df.columns = ["Image", "EncodedPixels"]
        df["Image"] = df["Image"].apply(lambda f: f.split("_")[0])
        df.set_index("Image", inplace=True)
        return df
