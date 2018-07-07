import torch
import torch.nn as nn

class VanillaModel(nn.Module):
    def __init__(self, num_hidden):
        super(VanillaModel, self).__init__()
        self.num_hidden = num_hidden
        self.name = 'vanilla_{}h'.format(num_hidden)
        self.sigmoid = nn.Sigmoid()
        self.main = nn.Sequential(
            nn.Linear(2,num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden,2),
            nn.ReLU(True),
            nn.Linear(2,1)
        )
    def forward(self, x):
        x = self.main(x)
        return self.sigmoid(x)

vanilla_2h = VanillaModel(2)
vanilla_3h = VanillaModel(3)

class ResidualModel(nn.Module):
    def __init__(self, num_hidden=1):
        super(ResidualModel, self).__init__()
        self.num_hidden = num_hidden
        self.name = 'residual_{}h'.format(num_hidden)
        self.sigmoid = nn.Sigmoid()
        self.main1 = nn.Sequential(
            nn.Linear(2, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, 2)
        )
        self.main2 = nn.Sequential(
            nn.Linear(2, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, 2)
        )
        self.linear = nn.Linear(2,1)

    def forward(self, x):
        x = self.main1(x) + x
        x = self.main2(x) + x
        x = self.linear(x)
        return self.sigmoid(x)

residual = ResidualModel()
