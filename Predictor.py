import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from ReaderWriter import read_data_lists
from VAE import data_loader, VAE


data_sequence_size = 5
batch_size = 5
convolution_channel_size_1 = 16
convolution_channel_size_2 = 8
convolution_channel_size_3 = 16
convolution_channel_size_4 = 4
fully_connected_unit_size = 400
latent_dimensions = 1
convolution_kernel = 3

latent_dimensions_thrust_torque = 1
weights_path_thrust = "./vae_net_thrust.pth"
weights_path_torque = "./vae_net_torque.pth"
seed = 1
lr = 1e-4
validation_data_fraction = 0.2
print_freq = 10


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


class Predictor(nn.Module):

    def __init__(self, z_dim):
        super(Predictor, self).__init__()

        # setup neural network
        vae_thrust = VAE(latent_dimensions_thrust_torque)
        vae_torque = VAE(latent_dimensions_thrust_torque)

        # load model weights
        vae_thrust.load_state_dict(torch.load(weights_path_thrust))
        vae_torque.load_state_dict(torch.load(weights_path_torque))

        # setup encoders
        self.encoder_thrust = vae_thrust.encoder
        self.encoder_torque = vae_torque.encoder

        self.decoder = Decoder(z_dim, convolution_channel_size_4 * data_sequence_size)

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        output = self.decoder(z_mean + eps * std)

        return output, z_mean, z_logvar

    def reconstruct_data(self, sample):
        return self.decoder(sample)


if __name__ == "__main__":
    # set seed
    torch.manual_seed(seed)

    # choose device(cpu or gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vae_thrust.to(device)
    vae_torque.to(device)

    # get data
    velocity_list, thrust_list, torque_list = read_data_lists()
    velocity_list = [x * 10 for x in velocity_list]
    thrust_list = [x / 10 for x in thrust_list]

    # split data into training and validation data
    velocity_train, velocity_val = train_test_split(velocity_list, test_size=validation_data_fraction,
                                                    train_size=1 - validation_data_fraction, shuffle=False)

    torque_train, torque_val = train_test_split(torque_list, test_size=validation_data_fraction,
                                                train_size=1 - validation_data_fraction, shuffle=False)

    thrust_train, thrust_val = train_test_split(thrust_list, test_size=validation_data_fraction,
                                                train_size=1 - validation_data_fraction, shuffle=False)

    # load training and validation data
    train_loader_velocity = data_loader(velocity_train, device)
    val_loader_velocity = data_loader(velocity_val, device)

    train_loader_torque = data_loader(torque_train, device)
    val_loader_torque = data_loader(torque_val, device)

    train_loader_thrust = data_loader(thrust_train, device)
    val_loader_thrust = data_loader(thrust_val, device)
