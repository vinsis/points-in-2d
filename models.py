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
vanilla_4h = VanillaModel(4)
vanilla_5h = VanillaModel(5)

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

residual_1h = ResidualModel()
residual_2h = ResidualModel(2)
residual_3h = ResidualModel(3)
residual_4h = ResidualModel(4)

class Residualv2(nn.Module):
    def __init__(self, num_hidden=1):
        super(Residualv2, self).__init__()
        self.num_hidden = num_hidden
        self.multiplier = nn.Parameter(torch.FloatTensor([1]))
        self.name = 'residual_v2_{}h'.format(num_hidden)
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
        x = self.main1(x) + (x * self.multiplier)
        x = self.main2(x) + (x * self.multiplier)
        x = self.linear(x)
        return self.sigmoid(x)

resnet_v2 = Residualv2()

# all_models = [vanilla_2h, vanilla_3h, vanilla_4h, vanilla_5h,
#     residual_1h, residual_2h, residual_3h, residual_4h, resnet_v2]

all_models = [residual_1h, resnet_v2]
