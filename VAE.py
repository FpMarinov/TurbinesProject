from sklearn.model_selection import train_test_split
import torch
import sys
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import plotly.express as px
import plotly.io as pio
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import Adam
from sklearn.manifold import TSNE
from trainer import Trainer
from DataReader import get_lists


# path = "./vae_net.pth"


class Encoder(nn.Module):

    def __init__(self, z_dim, fc1_size):
        super(Encoder, self).__init__()

        # 1 in channel, 64 out channels, 3 kernel, stride=padding=1
        self.conv1 = nn.Conv1d(1, 64, 3, 1, 1)
        self.bn_conv1 = nn.BatchNorm1d(64)

        # 64 in channel, 32 out channels, 3 kernel, stride=padding=1
        self.conv2 = nn.Conv1d(64, 32, 3, 1, 1)
        self.bn_conv2 = nn.BatchNorm1d(32)

        # kernel = 3, stride = 1, pooling
        self.pool = nn.MaxPool1d(3, 1)

        # fc1_size in, 400 out
        self.fc1 = nn.Linear(fc1_size, 400)
        self.bn_fc1 = nn.BatchNorm1d(400)

        # 400 in, z_dim out
        self.z_mu = nn.Linear(400, z_dim)
        self.z_sigma = nn.Linear(400, z_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn_conv1(x)

        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn_conv2(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        # x = self.bn_fc1(x)

        z_loc = self.z_mu(x)
        z_logvar = self.z_sigma(x)

        return z_loc, z_logvar


class Decoder(nn.Module):

    def __init__(self, z_dim, output_size):
        super(Decoder, self).__init__()

        # z_dim in, 400 out
        self.fc1 = nn.Linear(z_dim, 400)
        self.bn_fc1 = nn.BatchNorm1d(400)

        # 400 in, output_size out
        self.fc2 = nn.Linear(400, output_size)
        self.bn_fc2 = nn.BatchNorm1d(output_size)

        # 32 in channels, 64 out channels, 3 kernel, stride=padding=1
        self.conv1 = nn.Conv1d(32, 64, 3, 1, 1)
        self.bn_conv1 = nn.BatchNorm1d(64)

        # 64 in channels, 1 out channel, 3 kernel, stride=padding=1
        self.conv2 = nn.Conv1d(64, 1, 3, 1, 1)

    def forward(self, z_input):
        x = self.fc1(z_input)
        x = F.relu(x)
        # x = self.bn_fc1(x)

        x = self.fc2(x)
        x = F.relu(x)
        # x = self.bn_fc2(x)

        x = x.view(z_input.size()[0], 32, 1)
        x = F.interpolate(x, 3)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn_conv1(x)

        x = F.interpolate(x, 5)

        x = self.conv2(x)

        output = F.relu(x)

        return output


class VAE(nn.Module):

    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim, 32)
        self.decoder = Decoder(z_dim, 32)

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        output = self.decoder(z_mean + eps * std)

        return output, z_mean, z_logvar

    def reconstruct_digit(self, sample):
        return self.decoder(sample)


def loss_fn(output, mean, logvar, target):
    criterion = nn.MSELoss()
    mse = criterion(output, target)

    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return mse + kl


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MNIST VAE")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--z_dim", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--weights", type=str, default="./vae_net.pth")
    parser.add_argument("--visualise", type=bool, default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    vae = VAE(args.z_dim)
    optimizer = Adam(vae.parameters(), lr=args.lr)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    vae.to(device)

    velocity_list, thrust_list, torque_list = get_lists()

    data = velocity_list
    data_train, data_test = train_test_split(data, test_size=0.2, train_size=0.8, shuffle=False)

    tensor_train = torch.FloatTensor(data_train)
    tensor_train = tensor_train.to(device)
    train_dataset = TensorDataset(tensor_train)

    tensor_test = torch.FloatTensor(data_test)
    tensor_test = tensor_test.to(device)
    test_dataset = TensorDataset(tensor_test)

    # setup loaders
    if args.mode == "train":
        train_loader = DataLoader(train_dataset,
                                  batch_size=5, shuffle=True)
        val_loader = DataLoader(test_dataset,
                                batch_size=5, shuffle=True)

        trainer = Trainer(vae, args.epochs, train_loader, val_loader, device, loss_fn, optimizer, args.print_freq)
        trainer.train_model()
        torch.save(vae.state_dict(), args.weights)

    data_tensor = torch.FloatTensor(velocity_list)
    data_tensor = data_tensor.to(device)
    dataset = TensorDataset(data_tensor)

    val_loader = DataLoader(dataset,
                            batch_size=5, shuffle=True)

    if args.mode != "train":
        vae.load_state_dict(torch.load(args.weights))
        vae.eval()

    # if args.visualise:
    #
    #     latent_mnist = []
    #     target = []
    #     for dataset, targets in val_loader:
    #         z_means, z_logvar = vae.encoder(dataset)
    #         latent_mnist.extend(z_means.detach().numpy())
    #         target.extend(targets.numpy())
    #
    #     # take first 1k
    #     latent = np.array(latent_mnist)
    #     target = np.array(target)
    #     tsne = TSNE(n_components=2, init="pca", random_state=0)
    #
    #     X = tsne.fit_transform(latent)
    #
    #     dataset = np.vstack((X.T, target)).T
    #     df = pd.DataFrame(data=dataset, columns=["z1", "z2", "label"])
    #     df["label"] = df["label"].astype(str)
    #
    #     fig = px.scatter(df, x="z1", y="z2", color="label")
    #
    #     pio.write_html(fig, file="vis.html", auto_open=True)
