"""
Contains the variational autoencoder class, VAE, VAE's encoder class, Encoder, VAE's decoder class, Decoder,
and helper functions.

When the file is run, the behaviour is governed by the changeable parameters:
    - data_type (string): sets the data type, with which the program will work from among
        "velocity", "thrust" and "torque". The data is read from Data.txt.
    - mode (string): decides the mode of action of the program:
        - mode = "train", the variational autoencoder is trained with the chosen data type,
        saves its weights to a file with a name given by the weights_path variable, saves the training and
        validation losses to loss_record.csv, saves the training and validation mean squared errors to
        mse_loss_record.csv and produces plots of the losses and mean squared errors.
        - mode != "train"(or if mode = "train" and after the training is done), the variational autoencoder uses
        pretrained weights from a file with a name given by the weights_path variable to reconstruct
        the chosen data type and shows a scatter plot, having the original value on the x axis
        and the reconstructed value on the y axis.
    - epochs (int): decides the number of epochs the variational autoencoder trains for if it
        is in train mode.
    - plot_loss_1_epoch_skip (bool): decides whether to produce additional plots of the
        training and validation losses and mean squared errors, skipping the first epoch.
        It should only be used when the chosen number of epochs is 2 or more.
    - plot_loss_50_epoch_skip (bool): decides whether to produce additional plots of the
        training and validation losses and mean squared errors, skipping the first 50 epochs.
        It should only be used when the chosen number of epochs is 50 or more.
    - validation_data_fraction (float): the fraction of the data which is to be used for validation.

Based on:
    File Name: main.py
    Developed by: Nikolas Pitsillos, PhD Candidate in Computer Vision and Autonomous Systems @ UofG
    Taken from: https://github.com/npitsillos/mnist-vae/blob/master/main.py
    Described at: https://npitsillos.github.io/posts/2020/05/mnistvae/
"""

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

# variables that can be changed
data_type = "torque"
mode = "train"
epochs = 3
plot_loss_1_epoch_skip = True
plot_loss_50_epoch_skip = False
validation_data_fraction = 0.2

# variables that should not be changed
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


class Encoder(nn.Module):
    """
    Encoder for the Variational Autoencoder: VAE,
    and the Prediction Decoders: PredictionDecoder and PredictionDecoderVelocity.
    """

    def __init__(self, z_dim, fc1_size):
        """
        Initializes internal Encoder state.

        Args:
            z_dim (int): number of dimensions in latent space of VAE.
            fc1_size (int): size of the 1 dimensional input tensor, after
                flattening and before going through the fully connected
                transformation of Encoder.
        """
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
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): input tensor.

        Returns:
            tuple: (Tensor: gaussian means, Tensor: gaussian log variances).
        """
        # convolution
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)

        # reformat
        x = torch.flatten(x, 1)

        # fully connected transformation
        x = self.fc1(x)
        x = F.relu(x)

        # get mean and log variance
        z_loc = self.z_mu(x)
        z_logvar = self.z_sigma(x)

        return z_loc, z_logvar


class Decoder(nn.Module):
    """
    Decoder for the Variational Autoencoder: VAE.
    """

    def __init__(self, z_dim, output_size):
        """
        Initializes internal Decoder state.

        Args:
            z_dim (int): number of dimensions in latent space of VAE.
            output_size (int): size of the 1 dimensional output tensor, after going through the fully connected
                transformation of Decoder and before reformatting(changing dimensions).
        """
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
        """
        Defines the computation performed at every call.

        Args:
            z_input (Tensor): input tensor.

        Returns:
            Tensor: decoded input.
        """
        # latent space transformation
        x = self.fc_lat(z_input)
        x = F.relu(x)

        # fully connected transformation
        x = self.fc1(x)
        x = F.relu(x)

        # reformat
        x = x.view(z_input.size()[0], convolution_channel_size_4, data_sequence_size)

        # convolution
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)

        return x


class VAE(nn.Module):
    """
    Variational Autoencoder.
    """

    def __init__(self, z_dim):
        """
        Initializes internal VAE state.

        Args:
            z_dim (int): number of dimensions in latent space of VAE.
        """
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim, convolution_channel_size_4 * data_sequence_size)
        self.decoder = Decoder(z_dim, convolution_channel_size_4 * data_sequence_size)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): input tensor.

        Returns:
            tuple: (Tensor: output, Tensor: gaussian means, Tensor: gaussian log variances).
        """
        # encode inputs into means and logs of variances
        z_mean, z_logvar = self.encoder(x)

        # sample gaussians
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        sample = z_mean + eps * std

        # decode sample
        output = self.decoder(sample)

        return output, z_mean, z_logvar


def loss_fn(output, mean, logvar, target):
    """
    Loss function for VAE.

    Args:
        output (Tensor): output of VAE after encoding and then decoding.
        mean (Tensor): mean of gaussian distribution from latent space of VAE.
        logvar (Tensor): log variance of gaussian distribution from latent space of VAE.
        target (Tensor): input of VAE.

    Returns:
        tuple: (Tensor: total loss, Tensor: mean squared error).
    """
    # calculate the mean squared error per data point
    criterion = nn.MSELoss()
    mse = criterion(output, target)

    # calculate the mean kl divergence per data point
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    mkl = kl / (batch_size * data_sequence_size)

    return mse + mkl, mse


def data_loader(data, device, shuffle=True):
    """
    Data loader for VAE.

    Args:
        data (list): list of data points(floats).
        device (torch.device): torch device.
        shuffle (bool, optional): set to True to have the data reshuffled
            at every epoch (default: True).

    Returns:
        DataLoader: data loader.
    """
    # format data and transfer to device
    tensor = torch.FloatTensor(data).view(-1, 1, data_sequence_size)
    tensor = tensor.to(device)
    dataset = TensorDataset(tensor)

    # load data
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


def get_data():
    """
    Reads all data types from Data.txt and returns the data type specified by the data_type parameter as a list.

    Returns:
        list: a list of the data type specified by the data_type parameter.
    """
    # get all data
    velocity_list, thrust_list, torque_list = read_data_lists()

    # choose correct data
    if data_type == "velocity":
        data = velocity_list

        # multiply velocities by 10
        data = [x * 10 for x in data]
    elif data_type == "thrust":
        data = thrust_list

        # divide thrusts by 10
        data = [x / 10 for x in data]
    elif data_type == "torque":
        data = torque_list
    else:
        sys.exit("Incorrect data type.")

    return data


def setup(data):
    """
    Sets up the seed, the VAE neural network, the optimizer, the torch device,
    the splitting of the data into training and validation data sets,
    the loading of the training and validation data into data loaders, and the trainer.

    Args:
        data (list): list of data points(floats).

    Returns:
        tuple: (torch.device: torch device, VAE: neural network, VAETrainer: trainer).
    """
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
        average_total_training_losses, average_total_validation_losses, \
            average_mse_training_losses, average_mse_validation_losses = trainer.train_model()

        # save weights
        torch.save(vae.state_dict(), weights_path)

        # record average training and validation losses per epoch
        write_losses(average_total_training_losses, average_total_validation_losses,
                     average_mse_training_losses, average_mse_validation_losses)

        # visualise average training and validation losses per epoch
        losses_plot(average_total_training_losses, average_total_validation_losses,
                    plot_1_epoch_skip=plot_loss_1_epoch_skip, plot_50_epoch_skip=plot_loss_50_epoch_skip,
                    title="Total Training Loss")
        losses_plot(average_mse_training_losses, average_mse_validation_losses,
                    plot_1_epoch_skip=plot_loss_1_epoch_skip, plot_50_epoch_skip=plot_loss_50_epoch_skip,
                    title="MSE Training Loss")

    # load all data
    validation_loader = data_loader(data, device)

    # load model weights and activate evaluation if training is off
    if mode != "train":
        vae.load_state_dict(torch.load(weights_path))
        vae.eval()

    # visualise reconstruction
    reconstruction_scatter_plot(vae, validation_loader, data_type)

    plt.show()
