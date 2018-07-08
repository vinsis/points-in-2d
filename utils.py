import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import BCELoss
from loader import dataloader
from models import all_models
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

criterion = BCELoss()
device = ['cpu','cuda'][torch.cuda.is_available()]

optimizers, schedulers = {}, {}
for model in all_models:
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr = 0.002)
    optimizers[model.name] = optimizer

outer_limit = sum(dataloader.dataset.offset_and_length) + 1
points = product( np.arange(-outer_limit, outer_limit, 0.1), repeat=2 )
points = torch.FloatTensor(list(points)).to(device)

data = {}
data['points'] = points.cpu().numpy()

def save_boundary_labels(model, epoch, iteration, points=points):
    key = '{}_e{}_i{}'.format(model.name, epoch, iteration)
    with torch.no_grad():
        data[key] = model(points).cpu().numpy()

def save_obj(obj=data, filename='data.pkl'):
    with open(os.path.join('data', filename), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filename='data.pkl'):
    with open(os.path.join('data', filename), 'rb') as f:
        return pickle.load(f)

# def get_boundary_labels(model, points=points):
#     with torch.no_grad():
#         return model(points).cpu().numpy()
#
# def save_boundary_plot(model, epoch, iteration):
#     labels = get_boundary_labels(model)
#     # data = np.hstack([points.cpu().numpy(), labels])
#     filename = '{}_e{}_i{}.npy'.format(model.name, epoch, iteration)
#     filename = os.path.join('data', filename)
#     np.save(filename, labels)
#     # np.savetxt(filename, data, delimiter=',', fmt='%.3e')
#     # plt.scatter(data[:,0], data[:,1], c=data[:,2], s=1, cmap='viridis')
#     # plt.savefig(filename)
