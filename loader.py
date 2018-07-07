import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

two_pi = 2 * np.pi

class Point(Dataset):
    def __init__(self):
        self.offset_and_length = (3,2)
        self.inner_to_outer_ratio = 0.5

    def __len__(self):
        return 50000

    def to_outer_class(self, x):
        return self.offset_and_length[0] + x * self.offset_and_length[1]

    def polar_to_cartesian(self, r, theta):
        return torch.FloatTensor([r * torch.cos(theta), r * torch.sin(theta)])

    def __getitem__(self, index):
        torch.manual_seed(index)
        r = torch.rand(1).item()
        theta = torch.rand(1) * two_pi
        if r > self.inner_to_outer_ratio:
            return self.polar_to_cartesian(r, theta), 1
        r = self.to_outer_class(r)
        return self.polar_to_cartesian(r, theta), 0

dataloader = DataLoader(Point(), batch_size=32, shuffle=False)
