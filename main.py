from utils import all_models, optimizers, schedulers
from utils import criterion, dataloader, save_boundary_plot

def train(epoch):
    for i, (x,y) in enumerate(dataloader):
        y = y.unsqueeze(-1).float()
        for model in all_models:
            optimizer = optimizers[model.name]
            scheduler = schedulers[model.name]
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Iteration: [{}], Model: [{}], Loss: [{}]'.format(i, model.name, loss.item()))
            save_boundary_plot(model, epoch=epoch, iteration=i)
        # if i%1000 == 0:
            # scheduler.step()
    return i
