# Start with some standard imports.
import numpy as np
import torch
from torch.optim import Adam
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import Subset

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from src.models import FullyConvNet, ConvNet, MLP
from src.utils import train, evaluate, run_test, run_test_gradient

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),

    ])

    transform_pad = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Pad((23, 12, 31, 4))
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    ds_test = CIFAR10(root='./data', train=False, download=True, transform=transform)

    ds_test_pad = CIFAR10(root='./data', train=False, download=True, transform=transform)

    val_size = 5000
    batch_size = 512
    I = np.random.permutation(len(ds_train))
    ds_val = Subset(ds_train, I[:val_size])
    ds_train = Subset(ds_train, I[val_size:])

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size, num_workers=4, persistent_workers=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    dl_test_pad = torch.utils.data.DataLoader(ds_test_pad, batch_size, shuffle=True, num_workers=4,
                                              persistent_workers=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    dephts = [4, 16, 24, 36]

    ### Test with MLP
    model = MLP
    input_dim = 3072
    hidden_dim = 300
    run_test(dephts, model, "test_depth_MLP", loss_fn, dl_train, dl_test, dl_val, device, False, input_dim, hidden_dim,
             "MLP")
    run_test(dephts, model, "test_depth_MLP_res", loss_fn, dl_train, dl_test, dl_val, device, True, input_dim,
             hidden_dim, "MLP_Res")

    ### Test with ConvNet
    model = ConvNet
    input_dim = 3
    hidden_dim = 32
    run_test(dephts, model, "test_depth_conv", loss_fn, dl_train, dl_test, dl_val, device, False, input_dim, hidden_dim,
             "ConvNet")
    run_test(dephts, model, "test_depth_conv_res", loss_fn, dl_train, dl_test, dl_val, device, True, input_dim,
             hidden_dim, "ConvNet_Res")

    ### Test with Fully-convnet
    model = FullyConvNet
    input_dim = 3
    hidden_dim = 32
    run_test(dephts, model, "test_depth_fullyconv", loss_fn, dl_train, dl_test_pad, dl_val, device, False, input_dim,
             hidden_dim, "Fully-ConvNet")
    run_test(dephts, model, "test_depth_fullyconv_res", loss_fn, dl_train, dl_test_pad, dl_val, device, True, input_dim,
             hidden_dim, "Fully-ConvNet_Res")

    ### Test Gradient depth 5
    model_1 = ConvNet(input_dim=3, out_dim=10, hidden_dim=32, depth=5, residual_block=False).to(device)
    model_2 = ConvNet(input_dim=3, out_dim=10, hidden_dim=32, depth=5, residual_block=True).to(device)
    run_test_gradient(model_1, model_2, dl_train, loss_fn, device, "Convnet_depht_5", "Convnet_depht_5_residual",
                      "img/gradient")

    ### Test Gradient depth 10
    model_1 = ConvNet(input_dim=3, out_dim=10, hidden_dim=32, depth=10, residual_block=False).to(device)
    model_2 = ConvNet(input_dim=3, out_dim=10, hidden_dim=32, depth=10, residual_block=True).to(device)
    run_test_gradient(model_1, model_2, dl_train, loss_fn, device, "Convnet_depht_10", "Convnet_depht_10_residual",
                      "img/gradient")

    ### Test Gradient depth 20
    model_1 = ConvNet(input_dim=3, out_dim=10, hidden_dim=32, depth=20, residual_block=False).to(device)
    model_2 = ConvNet(input_dim=3, out_dim=10, hidden_dim=32, depth=20, residual_block=True).to(device)
    run_test_gradient(model_1, model_2, dl_train, loss_fn, device, "Convnet_depht_20", "Convnet_depht_20_residual",
                      "img/gradient")

    ### Test Gradient depth 40
    model_1 = ConvNet(input_dim=3, out_dim=10, hidden_dim=32, depth=40, residual_block=False).to(device)
    model_2 = ConvNet(input_dim=3, out_dim=10, hidden_dim=32, depth=40, residual_block=True).to(device)
    run_test_gradient(model_1, model_2, dl_train, loss_fn, device, "Convnet_depht_40", "Convnet_depht_40_residual",
                      "img/gradient")

