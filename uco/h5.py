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


# -- Writing raw predictions ----------------------------------------------------------


class HDF5PredictionWriterBase(HDF5ReaderWriterBase):
    """
    Handles writing prediction output to HDF5 by rounding to scaled uint8.
    """

    def __init__(self, filename, group_name, dataset_name):
        self.filename = Path(filename)
        self.group_name = group_name
        self.dataset_name = dataset_name
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.filename, "a")
        try:
            self.group = self.f.create_group(group_name)
        except ValueError:
            self.group = self.f[group_name]
        self.counter = 0

    def close(self):
        self.f.close()

    def summarise(self):
        return (
            f"Wrote {self.counter} predictions to "
            f'"{self.filename}" /{self.group_name}/{self.dataset_name}'
        )


class HDF5SegPredictionWriter(HDF5PredictionWriterBase):
    def __init__(self, filename, group_name, dataset_name, score):
        super().__init__(filename, group_name, dataset_name)
        try:
            self.dset = self.group.create_dataset(
                self.dataset_name, (self.N, self.C, self.H, self.W), dtype="uint8"
            )
        except Exception as _:  # noqa
            raise Exception("Skipping as predictions already recorded")
            self.dset = self.group[self.dataset_name]
        self.dset.attrs["score"] = score
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


class HDF5ClasPredictionWriter(HDF5PredictionWriterBase):
    def __init__(self, filename, group_name, dataset_name, score):
        super().__init__(filename, group_name, dataset_name)
        try:
            self.dset = self.group.create_dataset(
                self.dataset_name, (self.N, self.C), dtype="uint8"
            )
        except Exception as _:  # noqa
            self.dset = self.group[self.dataset_name]
        self.dset.attrs["score"] = score

    def write(self, data):
        rdata = self.wrangle_output(data)
        write_start = self.counter
        write_end = self.counter + rdata.shape[0]
        self.dset[write_start:write_end, :] = rdata
        self.counter += rdata.shape[0]

    def wrangle_output(self, data):
        return np.round(data * 100)


# -- Writing average predictions ------------------------------------------------------


class HDF5AverageWriterBase(HDF5ReaderWriterBase):
    """
    Handles averaging a set of predictions, and saving the result to HDF5.
    """

    dataset_name = "average"

    def __init__(self, verbose=2):
        self.logger = setup_logger(self, verbose)

    def average(self, pred_filename, avg_filename):
        self.logger.info(f'Reducing: "{pred_filename}" to "{avg_filename}"')
        self.average_groups(pred_filename, avg_filename)
        self.average_all(avg_filename)


class HDF5SegAverageWriterBase(HDF5AverageWriterBase):
    def __init__(self, verbose=2):
        super().__init__(verbose)

    def average_groups(self, pred_filename, avg_filename):
        with h5py.File(pred_filename, "r") as src, h5py.File(avg_filename, "w") as dst:
            for g in src.keys():
                scores = [src[g][k].attrs["score"] for k in src[g].keys()]
                weights = np.array([self.get_weight_model(s) for s in scores])
                weights = weights[:, np.newaxis, np.newaxis, np.newaxis]
                n_pred = weights.shape[0]
                self.logger.info(f"Averaging {n_pred} predictions from {g}")

                dset = dst.create_dataset(
                    g, (self.N, self.C, self.H, self.W), dtype="float32"
                )

                for n in tqdm(range(self.N), total=self.N):
                    pred_stack = np.stack(
                        [src[g][k][n, :, :, :] for k in src[g].keys()], axis=0
                    )
                    pred_stack = pred_stack * weights / 100  # undo scaling
                    pred_avg = np.sum(pred_stack, axis=0) / weights.sum()
                    # TODO: std calculation?
                    dset[n, :, :, :] = pred_avg
        self.logger.info(f"Finished averaging model groups.")

    def average_all(self, avg_filename):
        with h5py.File(avg_filename, "a") as h5:
            group_keys = [k for k in h5.keys() if k != self.dataset_name]
            weights = np.array([self.get_weight_group(k) for k in group_keys])
            weights = weights[:, np.newaxis, np.newaxis, np.newaxis]
            self.logger.info(f"Averaging {group_keys}")

            dset = h5.create_dataset(
                self.dataset_name, (self.N, self.C, self.H, self.W), dtype="float32"
            )

            for n in tqdm(range(self.N), total=self.N):
                pred_stack = np.stack([h5[k][n, :, :, :] for k in group_keys], axis=0)
                pred_stack = pred_stack * weights
                pred_avg = np.sum(pred_stack, axis=0) / weights.sum()
                # TODO: std calculation?
                dset[n, :, :, :] = pred_avg
        self.logger.info(f"Finished averaging all.")

    def get_weight_model(self, score):
        weight = max(0, score - 0.60)
        return weight

    def get_weight_group(self, name):
        weights = {
            "efficientnet-b0-FPN": 1.0,
            "efficientnet-b0-Unet": 0.5,
            "efficientnet-b2-FPN": 1.0,  # 0.6666
            "efficientnet-b2-Unet": 1.0,
            "efficientnet-b5-FPN": 1.0,  # 0.6665
            "efficientnet-b5-Unet": 0.5,
            "efficientnet-b6-FPN": 0.1,
            "resnext101_32x8d-FPN": 2.0,  # 0.6718
            "resnext101_32x8d-Unet": 1.0,
            "inceptionresnetv2-Unet": 0.1,
            "deeplabv3_resnet101-DeepLabV3": 0.5,  # 0.6651
        }
        w = weights.get(name, 0)
        return w


class HDF5ClasAverageWriterBase(HDF5AverageWriterBase):
    def __init__(self, verbose=2):
        super().__init__(verbose)

    def average_groups(self, pred_filename, avg_filename):
        with h5py.File(pred_filename, "r") as src, h5py.File(avg_filename, "w") as dst:
            for g in src.keys():
                scores = [src[g][k].attrs["score"] for k in src[g].keys()]
                weights = np.array([self.get_weight_model(s) for s in scores])
                weights = weights[:, np.newaxis]
                n_pred = weights.shape[0]
                self.logger.info(f"Averaging {n_pred} predictions from {g}")

                dset = dst.create_dataset(g, (self.N, self.C), dtype="float32")

                for n in tqdm(range(self.N), total=self.N):
                    pred_stack = np.stack(
                        [src[g][k][n, :] for k in src[g].keys()], axis=0
                    )
                    pred_stack = pred_stack * weights / 100  # undo scaling
                    pred_avg = np.sum(pred_stack, axis=0) / weights.sum()
                    # TODO: std calculation?
                    dset[n, :] = pred_avg
        self.logger.info(f"Finished averaging model groups.")

    def average_all(self, avg_filename):
        with h5py.File(avg_filename, "a") as h5:
            group_keys = [k for k in h5.keys() if k != self.dataset_name]
            weights = np.array([self.get_weight_group(k) for k in group_keys])
            weights = weights[:, np.newaxis]
            self.logger.info(f"Averaging {group_keys}")

            dset = h5.create_dataset(
                self.dataset_name, (self.N, self.C), dtype="float32"
            )

            for n in tqdm(range(self.N), total=self.N):
                pred_stack = np.stack([h5[k][n, :] for k in group_keys], axis=0)
                pred_stack = pred_stack * weights
                pred_avg = np.sum(pred_stack, axis=0) / weights.sum()
                # TODO: std calculation?
                dset[n, :] = pred_avg
        self.logger.info(f"Finished averaging all.")

    def get_weight_model(self, score):
        return 1

    def get_weight_group(self, group_name):
        return 1


# -- Post Processing ------------------------------------------------------------------


class PostProcessor(HDF5ReaderWriterBase):

    sample_csv = "sample_submission.csv"
    t0 = np.array([9573, 9670, 9019, 7885]) / 5
    c_factor = 9
    # top_thresholds = np.array([0.57, 0.57, 0.57, 0.57])
    top_thresholds = np.array([0.50, 0.50, 0.50, 0.50])
    bot_thresholds = np.array([0.40, 0.40, 0.40, 0.40])

    def __init__(self, verbose=2):
        self.logger = setup_logger(self, verbose)

    def process(self, seg_filename, clas_filename, data_dir, submission_filename):
        self.logger.info(f'PostProcessing: "{seg_filename}", "{clas_filename}"')
        self.logger.info(f"t0: {self.t0}")
        self.logger.info(f"c_factor: {self.c_factor}")
        self.logger.info(f"top_thresholds: {self.top_thresholds}")
        self.logger.info(f"bot_thresholds: {self.bot_thresholds}")
        sample_df = pd.read_csv(Path(data_dir) / self.sample_csv)

        throwaway_counter = np.array([0, 0, 0, 0])
        positive_counter = np.array([0, 0, 0, 0])
        dset_name = HDF5AverageWriterBase.dataset_name

        with h5py.File(seg_filename, "r") as f_seg, h5py.File(
            clas_filename, "r"
        ) as f_clas:
            for n in tqdm(range(self.N), total=self.N):
                pred_seg = f_seg[dset_name][n, :, :, :]
                pred_clas = f_clas[dset_name][n, :]
                rles, throwaways = self.threshold(pred_seg, pred_clas)
                throwaway_counter += throwaways
                for c, rle in enumerate(rles):
                    if rle != "":
                        positive_counter[c] += 1
                    sample_df.iloc[4 * n + c]["EncodedPixels"] = rle
        sample_df.to_csv(submission_filename, index=False)
        self.logger.info(f"Threw away {throwaway_counter} predictions under min size")
        self.logger.info(f"Positive predictions: {positive_counter}")
        self.logger.info(f'saved predictions to "{submission_filename}"')

    def threshold(self, pred_seg, pred_clas):
        """
        Post process predictions by applying triplet-threshold.
        """
        throwaways = np.array([0, 0, 0, 0])

        top_pass = (pred_seg > self.top_thresholds[:, np.newaxis, np.newaxis]).astype(
            np.uint8
        )

        min_sizes = compute_threshold(self.t0, self.c_factor, pred_clas)
        for c in range(self.C):
            if top_pass[c, :, :].sum() < min_sizes[c]:
                throwaways[c] += 1
                pred_seg[c, :, :] = 0

        bot_pass = (pred_seg > self.bot_thresholds[:, np.newaxis, np.newaxis]).astype(
            np.uint8
        )
        rles = [str(RLEOutput.from_mask(bot_pass[c, :, :])) for c in range(self.C)]
        return rles, throwaways


def compute_threshold(t0, c_factor, classification_output):
    """
    Adjust a threshold based on classification output.

    Parameters
    ----------
    t0 : numeric
        The original pixel threshold
    c_factor : numeric
        The amount a negative classification output will scale the pixel threshold.
    classification_output : numeric
        The output from a classifier in [0, 1]
    """
    return (t0 * c_factor) - (t0 * (c_factor - 1) * classification_output)


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
