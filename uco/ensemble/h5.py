from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import albumentations as A

from uco.data_loader import RLEOutput
from uco.utils import setup_logger, setup_logging


class HDF5ReaderWriterBase:

    H = 350
    W = 525
    C = 4
    N = 3698


class HDF5PredictionWriter(HDF5ReaderWriterBase):
    """
    Handles writing prediction output to HDF5 by rounding to scaled uint8.
    """

    def __init__(self, filename, dataset):
        self.filename = Path(filename)
        self.dataset = dataset
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.filename, 'a')
        try:
            self.dset = self.f.create_dataset(
                self.dataset,
                (self.N, self.C, self.H, self.W),
                # dtype='uint8',
                dtype='float32'
            )
        except Exception as _:  # noqa
            self.dset = self.f[self.dataset]
        self.counter = 0
        self.resizer = A.Resize(self.H, self.W, p=1)

    def write(self, data):
        rdata = self.wrangle_output(data)
        write_start = self.counter
        write_end = self.counter + rdata.shape[0]
        self.dset[write_start:write_end, :, :, :] = rdata
        self.counter += rdata.shape[0]

    def wrangle_output(self, data):
        resized_data = np.zeros((data.shape[0], self.C, self.H, self.W), dtype=np.float32)
        for n in range(data.shape[0]):
            for c in range(self.C):
                resized_data[n, c, :, :] = self.resizer(image=data[n, c, :, :])['image']
        # resized_data = np.round(resized_data * 100).astype(np.uint8)
        if (resized_data < 0).any():
            print(f'values below zero!')
        if (resized_data > 1).any():
            print(f'Values above 1!')
        if (resized_data.mean() > 0.8):
            print('Over 0.80 average!')
        return resized_data

    def close(self):
        self.f.close()

    def summarise(self):
        return f'Wrote {self.counter} predictions to "{self.filename}" /{self.dataset}'


class HDF5PredictionReducer(HDF5ReaderWriterBase):
    """
    Handles averaging a set of predictions.
    """

    sample_csv = 'sample_submission.csv'
    min_sizes = np.array([
        # 2nd percentile cutoff
        9573,
        9670,
        9019,
        7885,
    ])

    def __init__(self, verbose=2):
        setup_logging({'save_dir': 'saved', 'name': 'inference'})
        self.logger = setup_logger(self, verbose)

    def average(
        self,
        predictions_filename,
        data_dir,
        reduced_filename
    ):
        self.logger.info(f'Reducing: "{predictions_filename}"')
        data_dir = Path(data_dir)
        sample_df = pd.read_csv(data_dir / self.sample_csv)
        throwaway_counter = [0, 0, 0, 0]
        with h5py.File(predictions_filename, 'r') as f:
            for n in tqdm(range(self.N), total=self.N):
                pred_stack = np.stack([
                    f[k][n, :, :, :] for k in f.keys()
                ], axis=0)
                pred_mean = pred_stack.mean(axis=0) # / 100
                pred_bin = (pred_mean > 0.5).astype(np.uint8)
                for c in range(self.C):
                    # self.logger.debug(f'Checking: {pred_bin[c, :, :].sum()}')
                    if pred_bin[c, :, :].sum() < self.min_sizes[c]:
                        # self.logger.debug(f'throwing away {pred_bin[c, :, :].sum()} < {self.min_sizes[c]}')
                        throwaway_counter[c] += 1
                        pred_bin[c, :, :] = 0
                    rle = str(RLEOutput.from_mask(pred_bin[c, :, :]))
                    sample_df.iloc[4 * n + c]['EncodedPixels'] = rle
        pseudo_csv = data_dir / 'pseudo.csv'
        submission_csv = Path(predictions_filename).parent / reduced_filename
        sample_df.to_csv(pseudo_csv, index=False)
        sample_df.to_csv(submission_csv, index=False)
        self.logger.info(f'Threw away {throwaway_counter} predictions under min size')
        self.logger.info(f'Reduced predictions to "{pseudo_csv}" and "{submission_csv}"')
