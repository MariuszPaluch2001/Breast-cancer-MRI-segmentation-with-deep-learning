import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch

class MRIDataset(Dataset):
    def __init__(self, patients_dir = "./patients/ch1", class_dir = "./classes"):
        self.data_list = os.listdir(patients_dir)
        self.patients_dir = patients_dir
        self.class_dir = class_dir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = torch.load(f"{self.patients_dir}/{self.data_list[idx]}", weights_only=False).unsqueeze(0)
        mask = torch.load(f"{self.class_dir}/{self.data_list[idx]}", weights_only=False)
        return data, mask
