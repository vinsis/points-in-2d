import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import BCELoss
from loader import dataloader
from models import all_models
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import os

criterion = BCELoss()

optimizers, schedulers = {}, {}
for model in all_models:
    optimizer = Adam(model.parameters(), lr = 0.001)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma = 0.8)
    optimizers[model.name] = optimizer
    schedulers[model.name] = scheduler

outer_limit = sum(dataloader.dataset.offset_and_length) + 1
points = product( np.arange(-outer_limit, outer_limit, 0.1), repeat=2 )
points = torch.FloatTensor(list(points))

def get_boundary_labels(model, points=points):
    with torch.no_grad():
        return model(points).numpy()

def save_boundary_plot(model, epoch, iteration):
    labels = get_boundary_labels(model)
    data = np.hstack([points.numpy(), labels])
    filename = '{}_e{}_i{}.csv'.format(model.name, epoch, iteration)
    filename = os.path.join('data', filename)
    np.savetxt(filename, data, delimiter=',', fmt='%.3e')
    # plt.scatter(data[:,0], data[:,1], c=data[:,2], s=1, cmap='viridis')
    # plt.savefig(filename)
