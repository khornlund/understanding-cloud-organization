import numpy as np

IMAGE_ORIGINAL_HEIGHT = 1400
IMAGE_ORIGINAL_WIDTH = 2100
IMAGE_SUBMISSION_HEIGHT = 350
IMAGE_SUBMISSION_WIDTH = 525
N_CLASSES = 4


def make_mask(rle_encodings):
    shape = (IMAGE_ORIGINAL_HEIGHT, IMAGE_ORIGINAL_WIDTH, N_CLASSES)
    masks = np.zeros(shape, dtype=np.float32)
    for c, label in enumerate(rle_encodings.values):
        rle = RLEInput.from_str(label)
        masks[:, :, c] = rle.to_mask()
    return masks


class RLEBase:
    """
    Encapsulates run-length-encoding functionality.
    """

    MASK_H = 0
    MASK_W = 0

    @classmethod
    def from_str(cls, s):
        if s != s:
            return cls()
        list_ = [int(n) for n in s.split(" ")]
        return cls.from_list(list_)

    @classmethod
    def from_mask(cls, mask):
        pixels = mask.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return cls.from_list(runs)

    @classmethod
    def from_list(cls, list_):
        n_items = int(len(list_) / 2)
        items = np.zeros((n_items, 2), dtype=np.uint64)
        for i in range(n_items):
            items[i, 0] = list_[i * 2]
            items[i, 1] = list_[i * 2 + 1]
        return cls(items)

    def __init__(self, items=np.zeros((0, 0))):
        self._items = items

    @property
    def items(self):
        return self._items

    def __iter__(self):
        for idx, item in enumerate(self.items):
            yield (item[0], item[1])  # run, length

    def __len__(self):
        return self.items.shape[0]

    def to_mask(self):
        mask = np.zeros(self.MASK_H * self.MASK_W, dtype=np.uint8)
        for run, length in self:
            run = int(run - 1)
            end = int(run + length)
            mask[run:end] = 1
        return mask.reshape(self.MASK_H, self.MASK_W, order="F")

    def to_str_list(self):
        list_ = []
        for run, length in self:
            list_.append(str(run))
            list_.append(str(length))
        return list_

    def __str__(self):
        if len(self) == 0:
            return ""
        return " ".join(self.to_str_list())

    def __repr__(self):
        return self.__str__()


class RLEInput(RLEBase):

    MASK_H = IMAGE_ORIGINAL_HEIGHT
    MASK_W = IMAGE_ORIGINAL_WIDTH


class RLEOutput(RLEBase):

    MASK_H = IMAGE_SUBMISSION_HEIGHT
    MASK_W = IMAGE_SUBMISSION_WIDTH
