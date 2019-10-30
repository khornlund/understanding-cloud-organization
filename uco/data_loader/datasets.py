import cv2
from torch.utils.data import Dataset

from .process import make_mask, N_CLASSES


class CloudDatasetBase(Dataset):

    rle_cols = [f'rle{i}' for i in range(N_CLASSES)]
    bin_cols = [f'c{i}' for i in range(N_CLASSES)]

    @property
    def img_folder(self):
        raise NotImplementedError

    def __init__(self, df, data_dir, transforms):
        self.df = df
        self.data_dir = data_dir / self.img_folder
        self.transforms = transforms
        self.fnames = self.df.index.tolist()

    def read_rgb(self, idx):
        f = self.fnames[idx]
        img = cv2.imread(str(self.data_dir / f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return f, img

    def rle(self, idx):
        return self.df.iloc[idx][self.rle_cols]

    def binary(self, idx):
        return self.df.iloc[idx][self.bin_cols]

    def __len__(self):
        return len(self.fnames)


class CloudDatasetTrainVal(CloudDatasetBase):

    img_folder = 'train_images'

    def __init__(self, df, data_dir, transforms):
        super().__init__(df, data_dir, transforms)
        self.transforms = transforms

    def __getitem__(self, idx):
        mask = make_mask(self.rle(idx))
        _, img = self.read_rgb(idx)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        mask = mask.permute(2, 0, 1)  # HxWxC => CxHxW
        return img, mask


class CloudDatasetTest(CloudDatasetBase):

    img_folder = 'test_images'

    def __init__(self, df, data_dir, transforms):
        super().__init__(df, data_dir, transforms)
        self.transforms = transforms

    def __getitem__(self, idx):
        f, img = self.read_rgb(idx)
        augmented = self.transforms(image=img)
        img = augmented['image']
        return f, img


class CloudDatasetPseudo(CloudDatasetBase):

    img_folder = 'joined_images'

    def __init__(self, df, data_dir, transforms):
        super().__init__(df, data_dir, transforms)
        self.transforms = transforms

    def __getitem__(self, idx):
        mask = make_mask(self.rle(idx))
        _, img = self.read_rgb(idx)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        mask = mask.permute(2, 0, 1)  # HxWxC => CxHxW
        return img, mask
