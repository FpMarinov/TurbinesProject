from sklearn.model_selection import train_test_split
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from trainer import Trainer
from DataReader import get_lists
import matplotlib.pyplot as plt


data_type = "velocity"
mode = "test"
epochs = 11
visualise = True
drop_outliers = True

data_slice_size = 5
convolution_channel_size_1 = 64
convolution_channel_size_2 = 32
fully_connected_unit_size = 400
latent_dimensions = 20
convolution_kernel = 3
pooling_kernel = 3

weights_path = "./vae_net_%s.pth" % data_type

seed = 1
lr = 1e-3
print_freq = 10


class Encoder(nn.Module):

    def __init__(self, z_dim, fc1_size):
        super(Encoder, self).__init__()

        # 1 in channel, 64 out channels, 3 kernel, stride=padding=1
        self.conv1 = nn.Conv1d(1, convolution_channel_size_1, convolution_kernel, 1, 1)
        self.bn_conv1 = nn.BatchNorm1d(convolution_channel_size_1)

        # 64 in channel, 32 out channels, 3 kernel, stride=padding=1
        self.conv2 = nn.Conv1d(convolution_channel_size_1, convolution_channel_size_2, convolution_kernel, 1, 1)
        self.bn_conv2 = nn.BatchNorm1d(convolution_channel_size_2)

        # kernel = 3, stride = 1, pooling
        self.pool = nn.MaxPool1d(pooling_kernel, 1)

        # fc1_size in, 400 out
        self.fc1 = nn.Linear(fc1_size, fully_connected_unit_size)
        self.bn_fc1 = nn.BatchNorm1d(fully_connected_unit_size)

        # 400 in, z_dim out
        self.z_mu = nn.Linear(fully_connected_unit_size, z_dim)
        self.z_sigma = nn.Linear(fully_connected_unit_size, z_dim)

    def forward(self, x):
        # print("1:", x.size())
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn_conv1(x)
        # print("2:", x.size())
        x = self.pool(x)
        # print("3:", x.size())
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn_conv2(x)
        # print("4:", x.size())
        x = self.pool(x)
        # print("5:", x.size())
        x = torch.flatten(x, 1)
        # print("6:", x.size())
        x = self.fc1(x)
        x = F.relu(x)
        # print("7:", x.size())

        x = self.bn_fc1(x)
        # print("8:", x.size())

        z_loc = self.z_mu(x)
        z_logvar = self.z_sigma(x)

        return z_loc, z_logvar


class Decoder(nn.Module):

    def __init__(self, z_dim, output_size):
        super(Decoder, self).__init__()

        # z_dim in, 400 out
        self.fc1 = nn.Linear(z_dim, fully_connected_unit_size)
        self.bn_fc1 = nn.BatchNorm1d(fully_connected_unit_size)

        # 400 in, output_size out
        self.fc2 = nn.Linear(fully_connected_unit_size, output_size)
        self.bn_fc2 = nn.BatchNorm1d(output_size)

        # 32 in channels, 64 out channels, 3 kernel, stride=padding=1
        self.conv1 = nn.Conv1d(convolution_channel_size_2, convolution_channel_size_1, convolution_kernel, 1, 1)
        self.bn_conv1 = nn.BatchNorm1d(convolution_channel_size_1)

        # 64 in channels, 1 out channel, 3 kernel, stride=padding=1
        self.conv2 = nn.Conv1d(convolution_channel_size_1, 1, convolution_kernel, 1, 1)

    def forward(self, z_input):
        # print("-1:", z_input.size())
        x = self.fc1(z_input)
        x = F.relu(x)
        x = self.bn_fc1(x)
        # print("-2:", x.size())
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn_fc2(x)
        # print("-3:", x.size())
        x = x.view(z_input.size()[0], convolution_channel_size_2, 1)
        x = F.interpolate(x, 3)
        # print("-4:", x.size())
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn_conv1(x)
        # print("-5:", x.size())
        x = F.interpolate(x, 5)
        # print("-6:", x.size())
        x = self.conv2(x)
        # print("-7:", x.size())
        output = F.relu(x)
        # print("-8:", x.size())
        return output


class VAE(nn.Module):

    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim, convolution_channel_size_2)
        self.decoder = Decoder(z_dim, convolution_channel_size_2)

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
    # get data
    velocity_list, thrust_list, torque_list = get_lists()
    if data_type == "velocity":
        data = velocity_list
    elif data_type == "thrust":
        data = thrust_list
    elif data_type == "torque":
        data = torque_list
    else:
        sys.exit("Incorrect data_type.")

    # set seed
    torch.manual_seed(seed)

    # setup neural network and optimizer
    vae = VAE(latent_dimensions)
    optimizer = Adam(vae.parameters(), lr=lr)

    # choose device(cpu or gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vae.to(device)

    # split data into training and validation data
    data_train, data_val = train_test_split(data, test_size=0.2, train_size=0.8, shuffle=False)

    # format training data
    tensor_train = torch.FloatTensor(data_train).view(-1, 1, data_slice_size)
    print("init_train_size ", tensor_train.size())
    tensor_train = tensor_train.to(device)
    train_dataset = TensorDataset(tensor_train)

    # format validation data
    tensor_val = torch.FloatTensor(data_val).view(-1, 1, data_slice_size)
    print("init_val_size ", tensor_val.size())
    tensor_val = tensor_val.to(device)
    val_dataset = TensorDataset(tensor_val)

    # train model if training is on
    if mode == "train":
        # load training and validation data
        train_loader = DataLoader(train_dataset,
                                  batch_size=5, shuffle=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=5, shuffle=True)

        # do training
        trainer = Trainer(vae, epochs, train_loader, val_loader, device, loss_fn, optimizer, print_freq)
        trainer.train_model()

        # save weights
        torch.save(vae.state_dict(), weights_path)

    # format all data
    data_tensor = torch.FloatTensor(data).view(-1, 1, data_slice_size)
    data_tensor = data_tensor.to(device)
    dataset = TensorDataset(data_tensor)

    # load all data
    val_loader = DataLoader(dataset,
                            batch_size=5, shuffle=True)

    # test model if training is off
    if mode != "train":
        vae.load_state_dict(torch.load(weights_path))
        vae.eval()

    # visualise reconstruction if visualisation is on
    if visualise:
        reconstructions = []
        originals = []
        for inputs_targets in val_loader:

            inputs_targets = inputs_targets[0]

            outputs = vae(inputs_targets)

            outputs = outputs[0].detach().view(-1).to(torch.device('cpu'))
            outputs = outputs.numpy()
            reconstructions.extend(outputs)

            inputs_targets = inputs_targets.view(-1).to(torch.device('cpu'))
            inputs_targets = inputs_targets.numpy()
            originals.extend(inputs_targets)

        plt.scatter(originals, reconstructions)
        plt.ylabel("reconstruction")
        plt.xlabel("original")
        plt.title(data_type)

        # set up drop of outliers in visualisation if turned on
        if drop_outliers:
            y_upper_limit = max(data)
        else:
            y_upper_limit = None
        plt.ylim(top=y_upper_limit, bottom=0.0)

        plt.show()
