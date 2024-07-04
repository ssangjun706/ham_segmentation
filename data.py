import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torchvision.transforms as T

from PIL import Image
import numpy as np

from tqdm import tqdm
import pickle


def normalize(data):
    print("Normalizing Data...")
    mean = np.mean(data, axis=(2, 3), keepdims=True)
    std = np.std(data, axis=(2, 3), keepdims=True)
    standard_data = (data - mean) / std
    return standard_data


def transform(dir, name, size=(28, 28)):
    path = os.path.join(dir, name)
    image = Image.open(path)
    transform_image = image.resize(size)
    image.close()
    return np.array(transform_image)


def load_image(image_dir, mask_dir):
    print("Loading dataset...")
    IMG_PKL_PATH = os.path.join(image_dir, "HAM_IMAGE.pkl")
    MASK_PKL_PATH = os.path.join(mask_dir, "HAM_MASK.pkl")
    img, mask = None, None

    if not os.path.exists(IMG_PKL_PATH):
        img_list = [transform(image_dir, name) for name in tqdm(os.listdir(image_dir))]
        img_list = normalize(np.asarray(img_list).transpose(0, 3, 1, 2))
        with open(IMG_PKL_PATH, "wb") as fp:
            pickle.dump(img_list, fp)

    if not os.path.exists(MASK_PKL_PATH):
        mask_list = []
        for name in tqdm(os.listdir(mask_dir)):
            with Image.open(os.path.join(mask_dir, name)) as mask:
                mask_list.append(np.expand_dims(np.array(mask), axis=0))
        mask = np.asarray(mask_list)
        with open(MASK_PKL_PATH, "wb") as fp:
            pickle.dump(mask, fp)

    if img is None:
        img = pickle.load(open(IMG_PKL_PATH, "rb"))
    if mask is None:
        mask = pickle.load(open(MASK_PKL_PATH, "rb"))

    return img, mask


def train_val_test_split(dataset, ratio):
    train_size = len(dataset)
    test_size = int(ratio * train_size)
    train_size -= test_size
    train, test = random_split(dataset, [train_size, test_size])

    val_size = int(ratio * train_size)
    train_size -= val_size
    train, val = random_split(train, [train_size, val_size])
    return train, val, test


class HAM10000(Dataset):
    def __init__(self, image_dir, mask_dir):
        img, mask = load_image(image_dir, mask_dir)
        self.img = torch.tensor(img)
        self.mask = torch.tensor(mask)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx], self.mask[idx]
