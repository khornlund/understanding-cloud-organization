import numpy as np
from torch.utils.data.sampler import BatchSampler

from uco.utils import setup_logger


class SamplerFactory:
    """
    Factory class to create balanced samplers.
    """

    def __init__(self, verbose=0):
        self.logger = setup_logger(self, verbose)

    def get(self, class_idxs, batch_size, n_pseudo_per_batch):
        """
        Parameters
        ----------
        class_idxs : 2D list of ints
            List of sample indices for each class. Eg. [[0, 1], [2, 3]] implies
            indices 0, 1 belong to class 0, and indices 2, 3 belong to class 1.
        batch_size : int
            The batch size to use.
        n_pseudo_per_batch : int
            The number of pseudo-labelled samples to include in each batch.
        """
        self.logger.info(
            f"Creating `{str(WeightedFixedBatchSampler.__class__.__name__)}`..."
        )
        ground_truth_per_batch = batch_size - n_pseudo_per_batch
        n_ground_truth = len(class_idxs[0])
        n_batches = n_ground_truth // ground_truth_per_batch
        class_samples_per_batch = np.array([ground_truth_per_batch, n_pseudo_per_batch])
        self.logger.info(
            f"Expecting {class_samples_per_batch} samples of each class per batch, "
            f"over {n_batches} batches of size {batch_size}"
        )
        return WeightedFixedBatchSampler(class_samples_per_batch, class_idxs, n_batches)


class WeightedFixedBatchSampler(BatchSampler):
    """
    Ensures each batch contains a given class distribution.
    The lists of indices for each class are shuffled at the start of each call to
    `__iter__`.
    Parameters
    ----------
    class_samples_per_batch : `numpy.array(int)`
        The number of samples of each class to include in each batch.
    class_idxs : 2D list of ints
        The indices that correspond to samples of each class.
    n_batches : int
        The number of batches to yield.
    """

    def __init__(self, class_samples_per_batch, class_idxs, n_batches):
        self.class_samples_per_batch = class_samples_per_batch
        self.class_idxs = [CircularList(idx) for idx in class_idxs]
        self.n_batches = n_batches

        self.n_classes = len(self.class_samples_per_batch)
        self.batch_size = self.class_samples_per_batch.sum()

        assert len(self.class_samples_per_batch) == len(self.class_idxs)
        assert isinstance(self.n_batches, int)

    def _get_batch(self, start_idxs):
        selected = []
        for c, size in enumerate(self.class_samples_per_batch):
            selected.extend(self.class_idxs[c][start_idxs[c] : start_idxs[c] + size])
        np.random.shuffle(selected)
        return selected

    def __iter__(self):
        [cidx.shuffle() for cidx in self.class_idxs]
        start_idxs = np.zeros(self.n_classes, dtype=int)
        for bidx in range(self.n_batches):
            yield self._get_batch(start_idxs)
            start_idxs += self.class_samples_per_batch

    def __len__(self):
        return self.n_batches


class CircularList:
    """
    Applies modulo function to indexing.
    """

    def __init__(self, items):
        self._items = items
        self._mod = len(self._items)
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self._items)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(key.start, key.stop)]
        return self._items[key % self._mod]
