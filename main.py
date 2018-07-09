from utils import all_models, optimizers, device
from utils import criterion, dataloader, save_boundary_labels
from utils import save_obj, load_obj

def train(epoch):
    for i, (x,y) in enumerate(dataloader):
        y = y.unsqueeze(-1).float()
        x = x.to(device)
        y = y.to(device)
        for model in all_models:
            optimizer = optimizers[model.name]
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Epoch: [{}], Iteration: [{}], Model: [{}], Loss: [{}]'.format(epoch, i, model.name, loss.item()))
            if i % 4 == 0:
                save_boundary_labels(model, epoch=epoch, iteration=i//4)
    return i

if __name__ == '__main__':
    for epoch in range(1):
        train(epoch)
    save_obj()
