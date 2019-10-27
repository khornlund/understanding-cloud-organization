from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from uco.base import DataLoaderBase

from .datasets import CloudDatasetTrainVal, CloudDatasetTest


class CloudSegDataLoader(DataLoaderBase):

    train_csv = 'train.csv'
    test_csv  = 'sample_submission.csv'

    def __init__(
        self,
        transforms,
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        nworkers,
        pin_memory=True,
        train=True
    ):  # noqa
        self.transforms, self.shuffle = transforms, shuffle
        self.bs, self.nworkers, self.pin_memory = batch_size, nworkers, pin_memory
        self.data_dir = Path(data_dir)

        self.train_df, self.val_df = self.load_df(train, validation_split)

        tsfm = self.transforms.build(train=train)
        if train:
            dataset = CloudDatasetTrainVal(self.train_df, self.data_dir, tsfm)
        else:
            dataset = CloudDatasetTest(self.train_df, self.data_dir, tsfm)

        super().__init__(
            dataset,
            self.bs,
            shuffle=shuffle,
            num_workers=self.nworkers,
            pin_memory=self.pin_memory
        )

    def load_df(self, train, validation_split):
        csv_filename = self.train_csv if train else self.test_csv
        df = pd.read_csv(self.data_dir / csv_filename)
        df['Image'], df['Label'] = zip(*df['Image_Label'].str.split('_'))
        df = df.pivot(index='Image', columns='Label', values='EncodedPixels')
        df.columns = [f'rle{c}' for c in range(4)]
        df['n_classes'] = df.count(axis=1)

        # add classification columns
        for c in range(4):
            df[f'c{c}'] = df[f'rle{c}'].apply(lambda rle: not pd.isnull(rle))

        if train and validation_split > 0:
            return train_test_split(df, test_size=validation_split, stratify=df["n_classes"])

        return df, pd.DataFrame({})

    def split_validation(self):
        if self.val_df.empty:
            return None
        else:
            tsfm = self.transforms.build(train=False)
            dataset = CloudDatasetTrainVal(self.val_df, self.data_dir, tsfm)
            return DataLoader(
                dataset,
                self.bs // 8,
                num_workers=self.nworkers,
                pin_memory=self.pin_memory
            )
