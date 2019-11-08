from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import albumentations as A

from uco.data_loader import RLEOutput
from uco.utils import setup_logger


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
        self.f = h5py.File(self.filename, "a")
        try:
            self.dset = self.f.create_dataset(
                self.dataset,
                (self.N, self.C, self.H, self.W),
                dtype="uint8",
                # dtype='float32'
            )
        except Exception as _:  # noqa
            self.dset = self.f[self.dataset]
        self.counter = 0
        self.resizer = A.Resize(self.H, self.W, interpolation=cv2.INTER_CUBIC, p=1)

    def write(self, data):
        rdata = self.wrangle_output(data)
        write_start = self.counter
        write_end = self.counter + rdata.shape[0]
        self.dset[write_start:write_end, :, :, :] = rdata
        self.counter += rdata.shape[0]

    def wrangle_output(self, data):
        resized_data = np.zeros(
            (data.shape[0], self.C, self.H, self.W), dtype=np.float32
        )
        for n in range(data.shape[0]):
            for c in range(self.C):
                resized_data[n, c, :, :] = self.resizer(image=data[n, c, :, :])["image"]
        return np.round(resized_data * 100)

    def close(self):
        self.f.close()

    def summarise(self):
        return f'Wrote {self.counter} predictions to "{self.filename}" /{self.dataset}'


class HDF5AverageWriter(HDF5ReaderWriterBase):
    """
    Handles averaging a set of predictions, and saving the result to HDF5.
    """

    dataset_name = "average"

    def __init__(self, verbose=2):
        self.logger = setup_logger(self, verbose)

    def average(self, pred_filename, avg_filename):
        self.logger.info(f'Reducing: "{pred_filename}" to "{avg_filename}"')
        with h5py.File(pred_filename, "r") as src, h5py.File(avg_filename, "w") as dst:
            n_predictions = sum([1 for _ in src.keys()])
            self.logger.info(f"Averaging {n_predictions} predictions")

            dset = dst.create_dataset(
                self.dataset_name, (self.N, self.C, self.H, self.W), dtype="float32"
            )

            for n in tqdm(range(self.N), total=self.N):
                pred_stack = np.stack([src[k][n, :, :, :] for k in src.keys()], axis=0)
                pred_mean = pred_stack.mean(axis=0) / 100  # undo scaling
                # TODO: std calculation?
                dset[n, :, :, :] = pred_mean

        self.logger.info(f"Averaging complete.")


class PostProcessor(HDF5ReaderWriterBase):

    sample_csv = "sample_submission.csv"
    min_sizes = np.array([9573, 9670, 9019, 7885])
    top_thresholds = np.array([0.50, 0.50, 0.50, 0.50])
    bot_thresholds = np.array([0.50, 0.50, 0.50, 0.50])

    def __init__(self, verbose=2):
        self.logger = setup_logger(self, verbose)

    def process(self, avg_filename, data_dir, submission_filename):
        self.logger.info(f'PostProcessing: "{avg_filename}"')
        self.logger.info(f"min_sizes: {self.min_sizes}")
        self.logger.info(f"top_thresholds: {self.top_thresholds}")
        self.logger.info(f"bot_thresholds: {self.bot_thresholds}")
        sample_df = pd.read_csv(Path(data_dir) / self.sample_csv)

        throwaway_counter = np.array([0, 0, 0, 0])
        positive_counter = np.array([0, 0, 0, 0])
        dset_name = HDF5AverageWriter.dataset_name

        with h5py.File(avg_filename, "r") as f:
            for n in tqdm(range(self.N), total=self.N):
                pred_mean = f[dset_name][n, :, :, :]
                rles, throwaways = self.threshold(pred_mean)
                throwaway_counter += throwaways
                for c, rle in enumerate(rles):
                    if rle != "":
                        positive_counter[c] += 1
                    sample_df.iloc[4 * n + c]["EncodedPixels"] = rle
        sample_df.to_csv(submission_filename, index=False)
        self.logger.info(f"Threw away {throwaway_counter} predictions under min size")
        self.logger.info(f"Positive predictions: {positive_counter}")
        self.logger.info(f'saved predictions to "{submission_filename}"')

    def threshold(self, predictions):
        """
        Post process predictions by applying triplet-threshold.
        """
        throwaways = np.array([0, 0, 0, 0])

        top_pass = (
            predictions > self.top_thresholds[:, np.newaxis, np.newaxis]
        ).astype(np.uint8)
        for c in range(self.C):
            if top_pass[c, :, :].sum() < self.min_sizes[c]:
                throwaways[c] += 1
                predictions[c, :, :] = 0

        bot_pass = (
            predictions > self.bot_thresholds[:, np.newaxis, np.newaxis]
        ).astype(np.uint8)
        rles = [str(RLEOutput.from_mask(bot_pass[c, :, :])) for c in range(self.C)]
        return rles, throwaways


def draw_convex_hull(mask, mode="convex"):
    """
    https://www.kaggle.com/ratthachat/cloud-convexhull-polygon-postprocessing-no-gpu
    """
    img = np.zeros(mask.shape)
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if mode == "rect":  # simple rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        elif mode == "convex":  # minimum convex hull
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, (255, 255, 255), -1)
        elif mode == "approx":
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(img, [approx], 0, (255, 255, 255), -1)
        else:  # minimum area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 255, 255), -1)
    return img / 255.0
