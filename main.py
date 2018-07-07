from loader import dataloader
from utils import vanilla_2h, vanilla_3h, residual, optimizers, criterion

models = [vanilla_2h, vanilla_3h, residual]
def train():
    for i, (x,y) in enumerate(dataloader):
        y = y.unsqueeze(-1).float()
        for model in models:
            optimizer = optimizers[model.name]
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print('Iteration: [{}], Model: [{}], Loss: [{}]'.format(i, model.name, loss.item()))
        if i%1000 == 0:
            print('\n')
    return i
