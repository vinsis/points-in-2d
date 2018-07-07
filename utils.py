from torch.optim import SGD
from torch.nn import BCELoss
from models import vanilla_2h, vanilla_3h, residual

criterion = BCELoss()

optimizers = {}
for model in [vanilla_2h, vanilla_3h, residual]:
    optimizer = SGD(model.parameters(), lr = 0.01, momentum = 0.9)
    optimizers[model.name] = optimizer
