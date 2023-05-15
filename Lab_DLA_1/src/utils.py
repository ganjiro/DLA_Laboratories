import os.path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sb


def train(model, data_loader, epoch, loss_fn, optimizer, device):
    tot_loss = 0
    for (images, labels) in tqdm(data_loader, desc=f'Training epoch {epoch}', leave=False):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        loss = loss_fn(logits, labels)
        tot_loss += loss
        loss.backward()
        optimizer.step()

    return tot_loss / len(data_loader)


def evaluate(model, data_loader, epoch, device):
    tot_acc = 0
    for (images, labels) in tqdm(data_loader, desc=f'Test epoch {epoch}', leave=False):
        images = images.to(device)

        with torch.no_grad():
            logits = model(images).cpu().numpy()
        predictions = np.argmax(logits, axis=1)

        accuracy = np.count_nonzero(np.equal(predictions, labels)) / len(predictions)
        tot_acc += accuracy

    return tot_acc / len(data_loader)


def run_test(dephts, model_class, writer_name, loss_fn, dl_train, dl_test, dl_val, device, residual, input_dim,
             hidden_dim, label):
    res_accuracy_test = []
    res_accuracy_val = []
    res_traing_loss = []
    for depht in dephts:
        accuracy_test = []
        accuracy_val = []
        traing_loss = []

        model = model_class(input_dim=input_dim, out_dim=10, hidden_dim=hidden_dim, depth=depht,
                            residual_block=residual).to(device)

        optimizer = Adam(model.parameters(), lr=0.001)

        writer = SummaryWriter(writer_name + "/depht_" + str(depht))

        for epoch in range(50):
            loss = train(model, dl_train, epoch, loss_fn, optimizer, device)
            writer.add_scalar("Train/loss", loss, epoch)
            traing_loss.append(loss.cpu().item())

            if epoch % 5 == 0:
                accuracy = evaluate(model, dl_val, epoch, device)
                writer.add_scalar("Train/validation", accuracy, epoch)
                accuracy_val.append(accuracy)

        test_accuracy = evaluate(model, dl_test, 50, device)
        writer.add_scalar("Test/Accuracy", test_accuracy, 0)
        accuracy_test.append(test_accuracy)

        writer.flush()
        writer.close()

        res_accuracy_test.append(accuracy_test)
        res_accuracy_val.append(accuracy_val)
        res_traing_loss.append(traing_loss)

    plot(dephts, label, writer_name, res_accuracy_test, res_accuracy_val, res_traing_loss)


def plot(dephts, label, path, res_accuracy_test, res_accuracy_val, res_traing_loss):
    plt_val = plt.figure(1)
    plt_train = plt.figure(2)
    plt_test = plt.figure(3)

    x_train = [i for i in range(50)]
    x_val = [i for i in range(0, 50, 5)]
    plt.figure(3)
    plt.plot(dephts, res_accuracy_test)

    for i, depth in enumerate(dephts):
        plt.figure(2)
        plt.plot(x_train, res_traing_loss[i], label=label + "_" + str(depth))
        plt.figure(1)
        plt.plot(x_val, res_accuracy_val[i], label=label + "_" + str(depth))

    plt.figure(1)
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(path, "Validation_" + label))
    plt.close()

    plt.figure(2)
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(path, "Train_" + label))
    plt.close()

    plt.figure(3)
    plt.ylabel("Accuracy")
    plt.xlabel("Depth")
    plt.savefig(os.path.join(path, "Test_" + label))
    plt.close()


def run_test_gradient(model_1, model_2, dl_train, loss_fn, device, label_1, label_2, path):
    optimizer_1 = Adam(model_1.parameters(), lr=0.001)
    optimizer_2 = Adam(model_2.parameters(), lr=0.001)

    for epoch in range(2):
        train(model_1, dl_train, epoch, loss_fn, optimizer_1, device)
        train(model_2, dl_train, epoch, loss_fn, optimizer_2, device)
    plot_gradient_heatmap(model_1, label_1 + "_epoch_2", path)
    plot_gradient_heatmap(model_2, label_2 + "_epoch_2", path)

    for epoch in range(28):
        train(model_1, dl_train, epoch, loss_fn, optimizer_1, device)
        train(model_2, dl_train, epoch, loss_fn, optimizer_2, device)
    plot_gradient_heatmap(model_1, label_1 + "_epoch_30", path)
    plot_gradient_heatmap(model_2, label_2 + "_epoch_30", path)


def plot_gradient_heatmap(model, label, path):
    grad = model.layer_in.weight.grad[0].detach().cpu().numpy()

    sb.heatmap(grad.mean(0), xticklabels=False, yticklabels=False, annot=True, cbar=False)
    plt.title("Gradient " + label)
    plt.savefig(os.path.join(path, label))
    plt.close()
