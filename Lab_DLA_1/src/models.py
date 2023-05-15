from builtins import int

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, depth=3, residual_block=False):
        super().__init__()
        self.residual_block = residual_block
        self.layer_in = nn.Linear(input_dim, hidden_dim)

        self.fc_deep = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 2)])

        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        out = x.flatten(1)

        out = F.relu(self.layer_in(out))

        for layer in self.fc_deep:
            next = F.relu(layer(out))

            if self.residual_block:
                out = next + out
            else:
                out = next

        return self.fc_out(out)


class ConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, depth=3, residual_block=False):
        super().__init__()

        self.residual_block = residual_block
        self.layer_in = nn.Conv2d(in_channels=input_dim, kernel_size=3, out_channels=hidden_dim, stride=2, padding=1)

        self.conv_deep = nn.ModuleList(
            [nn.Conv2d(in_channels=hidden_dim, kernel_size=3, out_channels=hidden_dim, padding=1) for _ in range(depth - 2)])

        self.conv_out = nn.Conv2d(in_channels=hidden_dim, kernel_size=3, out_channels=int(hidden_dim), stride=2, padding=1)
        self.fc_out = nn.Linear(in_features=128, out_features=out_dim)
        self.max_pool = nn.MaxPool2d(3, stride=3)

    def forward(self, x):
        out = F.relu(self.layer_in(x))

        for layer in self.conv_deep:
            next = F.relu(layer(out))

            if self.residual_block:
                out = next + out
            else:
                out = next

        out = F.relu(self.conv_out(out))
        out = self.max_pool(out)
        out = out.flatten(1)
        return self.fc_out(out)



class FullyConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, depth=3, residual_block=False):
        super().__init__()
        self.residual_block = residual_block

        self.layer_in = nn.Conv2d(in_channels=input_dim, kernel_size=3, out_channels=hidden_dim, stride=2, padding=1)

        self.conv_deep = nn.ModuleList(
            [nn.Conv2d(in_channels=hidden_dim, kernel_size=3, out_channels=hidden_dim, padding=1) for _ in
             range(depth - 2)])

        self.conv_out = nn.Conv2d(in_channels=hidden_dim, kernel_size=3, out_channels=int(hidden_dim), stride=2,
                                  padding=1)
        self.conv_1x1 = nn.Conv2d(in_channels=hidden_dim, kernel_size=1, out_channels=out_dim)


    def forward(self, x):
        out = F.relu(self.layer_in(x))

        for layer in self.conv_deep:
            next = F.relu(layer(out))

            if self.residual_block:
                out = next + out
            else:
                out = next

        out = F.relu(self.conv_out(out))
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        out = self.conv_1x1(out)
        return out.flatten(1)