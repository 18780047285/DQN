import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, n_actions)
        self.fc1.weight.data.normal_(0.5, 0.1)
        self.fc2.weight.data.normal_(0.5, 0.1)
        self.fc3.weight.data.normal_(-10, 1)


    #     self.conv = nn.Sequential(
    #         nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
    #         nn.ReLU(),
    #         nn.Conv2d(32, 64, kernel_size=4, stride=2),
    #         nn.ReLU(),
    #         nn.Conv2d(64, 64, kernel_size=3, stride=1),
    #         nn.ReLU()
    #     )

    #     conv_out_size = self._get_conv_out(input_shape)
    #     self.fc = nn.Sequential(
    #         nn.Linear(conv_out_size, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, n_actions)
    #     )

    # def _get_conv_out(self, shape):
    #     o = self.conv(torch.zeros(1, *shape))
    #     return int(np.prod(o.size()))

    def forward(self, x):
        # conv_out = self.conv(x).view(x.size()[0], -1)
        # return self.fc(conv_out)
        x = x.float().requires_grad_(True)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)
        return out
