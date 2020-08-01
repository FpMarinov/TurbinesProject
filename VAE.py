from sklearn.model_selection import train_test_split
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import matplotlib.pyplot as plt
from VAETrainer import VAETrainer
from ReaderWriter import read_data_lists, write_losses
from Plotter import losses_plot, reconstruction_scatter_plot


data_type = "velocity"
mode = "train"
epochs = 100
plot_loss_50_epoch_skip = True

data_sequence_size = 5
batch_size = 5
convolution_channel_size_1 = 16
convolution_channel_size_2 = 8
convolution_channel_size_3 = 16
convolution_channel_size_4 = 4
fully_connected_unit_size = 400
latent_dimensions = 1
convolution_kernel = 3
weights_path = "./vae_net_%s.pth" % data_type
lr = 1e-4
seed = 1
validation_data_fraction = 0.2


class Encoder(nn.Module):

    def __init__(self, z_dim, fc1_size):
        super(Encoder, self).__init__()

        # convolution
        self.conv1 = nn.Conv1d(1, convolution_channel_size_1, convolution_kernel, 1, 1)
        self.conv2 = nn.Conv1d(convolution_channel_size_1, convolution_channel_size_2, convolution_kernel, 1, 1)
        self.conv3 = nn.Conv1d(convolution_channel_size_2, convolution_channel_size_3, convolution_kernel, 1, 1)
        self.conv4 = nn.Conv1d(convolution_channel_size_3, convolution_channel_size_4, convolution_kernel, 1, 1)

        # fully connected transformation
        self.fc1 = nn.Linear(fc1_size, fully_connected_unit_size)

        # latent space transformation
        self.z_mu = nn.Linear(fully_connected_unit_size, z_dim)
        self.z_sigma = nn.Linear(fully_connected_unit_size, z_dim)

    def forward(self, x):
        # print("1:", x.size())

        x = self.conv1(x)
        x = F.relu(x)
        # print("2:", x.size())

        x = self.conv2(x)
        x = F.relu(x)
        # print("3:", x.size())

        x = self.conv3(x)
        x = F.relu(x)
        # print("4:", x.size())

        x = self.conv4(x)
        x = F.relu(x)
        # print("5:", x.size())

        x = torch.flatten(x, 1)
        # print("6:", x.size())

        x = self.fc1(x)
        x = F.relu(x)
        # print("7:", x.size())

        z_loc = self.z_mu(x)
        z_logvar = self.z_sigma(x)
        # print("8:", z_loc.size())

        return z_loc, z_logvar


class Decoder(nn.Module):

    def __init__(self, z_dim, output_size):
        super(Decoder, self).__init__()

        # latent space transformation
        self.fc_lat = nn.Linear(z_dim, fully_connected_unit_size)

        # fully connected transformation
        self.fc1 = nn.Linear(fully_connected_unit_size, output_size)

        # convolution
        self.conv1 = nn.Conv1d(convolution_channel_size_4, convolution_channel_size_3, convolution_kernel, 1, 1)
        self.conv2 = nn.Conv1d(convolution_channel_size_3, convolution_channel_size_2, convolution_kernel, 1, 1)
        self.conv3 = nn.Conv1d(convolution_channel_size_2, convolution_channel_size_1, convolution_kernel, 1, 1)
        self.conv4 = nn.Conv1d(convolution_channel_size_1, 1, convolution_kernel, 1, 1)

    def forward(self, z_input):
        # print("-1:", z_input.size())

        x = self.fc_lat(z_input)
        x = F.relu(x)
        # print("-2:", x.size())

        x = self.fc1(x)
        x = F.relu(x)
        # print("-3:", x.size())

        x = x.view(z_input.size()[0], convolution_channel_size_4, data_sequence_size)
        # print("-4:", x.size())

        x = self.conv1(x)
        x = F.relu(x)
        # print("-5:", x.size())

        x = self.conv2(x)
        x = F.relu(x)
        # print("-6:", x.size())

        x = self.conv3(x)
        x = F.relu(x)
        # print("-7:", x.size())

        x = self.conv4(x)
        output = F.relu(x)
        # print("-8:", x.size())

        return output


class VAE(nn.Module):

    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim, convolution_channel_size_4 * data_sequence_size)
        self.decoder = Decoder(z_dim, convolution_channel_size_4 * data_sequence_size)

    def forward(self, x):
        # encode inputs into means and logs of variances
        z_mean, z_logvar = self.encoder(x)

        # sample gaussians
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        sample = z_mean + eps * std

        # decode sample
        output = self.decoder(sample)

        return output, z_mean, z_logvar

    def reconstruct_data(self, sample):
        return self.decoder(sample)


def loss_fn(output, mean, logvar, target):
    # calculate the mean squared error per data point
    criterion = nn.MSELoss()
    mse = criterion(output, target)

    # calculate the mean kl divergence per data point
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    mkl = kl / (batch_size * data_sequence_size)

    return mse + mkl


def data_loader(data, device, shuffle=True):
    # format data
    tensor = torch.FloatTensor(data).view(-1, 1, data_sequence_size)
    tensor = tensor.to(device)
    dataset = TensorDataset(tensor)

    # load data
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


def get_data():
    # get all data
    velocity_list, thrust_list, torque_list = read_data_lists()

    # choose correct data
    if data_type == "velocity":
        data = velocity_list
        data = [x * 10 for x in data]
    elif data_type == "thrust":
        data = thrust_list
        data = [x / 10 for x in data]
    elif data_type == "torque":
        data = torque_list
    else:
        sys.exit("Incorrect data type.")

    return data


def setup(data):
    # set seed
    torch.manual_seed(seed)

    # setup neural network and optimizer
    vae = VAE(latent_dimensions)
    optimizer = Adam(vae.parameters(), lr=lr)

    # choose device(cpu or gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vae.to(device)

    if mode == "train":
        # split data into training and validation data
        training_data, validation_data = train_test_split(data, test_size=validation_data_fraction,
                                                          train_size=1 - validation_data_fraction, shuffle=False)
        # load training and validation data
        training_loader = data_loader(training_data, device)
        validation_loader = data_loader(validation_data, device)

        trainer = VAETrainer(vae, epochs, training_loader, validation_loader, device, loss_fn, optimizer)
    else:
        trainer = None

    return device, vae, trainer


if __name__ == "__main__":
    # get data
    data = get_data()

    # setup
    device, vae, trainer = setup(data)

    # train model if training is on
    if mode == "train":
        # do training and get losses
        average_training_losses, average_validation_losses = trainer.train_model()

        # save weights
        torch.save(vae.state_dict(), weights_path)

        # record average training and validation losses per epoch
        write_losses(average_training_losses, average_validation_losses)

        # visualise average training and validation losses per epoch
        losses_plot(average_training_losses, average_validation_losses, plot_loss_50_epoch_skip)

    # load all data
    validation_loader = data_loader(data, device)

    # load model weights and activate evaluation if training is off
    if mode != "train":
        vae.load_state_dict(torch.load(weights_path))
        vae.eval()

    # visualise reconstruction
    reconstruction_scatter_plot(vae, data, validation_loader, data_type)

    plt.show()
