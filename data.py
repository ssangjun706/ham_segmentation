import os
from torch.utils.data import Dataset


class HAM10000(Dataset):
    def __init__(self, dir):
        self.files = os.listdir(dir)
        print(len(self.files))
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
