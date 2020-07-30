import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from Plotter import losses_plot, prediction_reconstruction_scatter_plot
from PredictionTrainer import PredictionTrainer
from ReaderWriter import read_data_lists, write_losses
from VAE import data_loader, VAE, latent_dimensions, data_sequence_size, seed, validation_data_fraction

mode = "train"
epochs = 1
visualise_scatter = True
show_y_equals_x = True
visualise_training_and_validation_loss = True
plot_loss_50_epoch_skip = False

convolution_channel_size_1 = 16
convolution_channel_size_2 = 8
convolution_channel_size_3 = 16
convolution_channel_size_4 = 4
fully_connected_unit_size = 400
convolution_kernel = 3

weights_path_thrust = "./vae_net_thrust.pth"
weights_path_torque = "./vae_net_torque.pth"
weights_path_decoder = "./vae_net_prediction_decoder.pth"
lr = 1e-4
print_freq = 10


class PredictionDecoder(nn.Module):

    def __init__(self):
        super(PredictionDecoder, self).__init__()

        # latent space transformation
        self.fc_lat = nn.Linear(4 * latent_dimensions, fully_connected_unit_size)

        # fully connected transformation
        self.fc1 = nn.Linear(fully_connected_unit_size, convolution_channel_size_4 * data_sequence_size)

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


if __name__ == "__main__":
    # set seed
    torch.manual_seed(seed)

    # setup neural networks and optimizer
    vae_thrust = VAE(latent_dimensions)
    vae_torque = VAE(latent_dimensions)
    decoder = PredictionDecoder()
    optimizer = Adam(decoder.parameters(), lr=lr)

    # load model weights
    vae_thrust.load_state_dict(torch.load(weights_path_thrust))
    vae_torque.load_state_dict(torch.load(weights_path_torque))

    # get encoders
    encoder_thrust = vae_thrust.encoder
    encoder_torque = vae_torque.encoder

    # set encoders to evaluation
    encoder_thrust.eval()
    encoder_torque.eval()

    # choose device(cpu or gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoder_thrust.to(device)
    encoder_torque.to(device)

    # get data
    velocity_list, thrust_list, torque_list = read_data_lists()
    velocity_list = [x * 10 for x in velocity_list]
    thrust_list = [x / 10 for x in thrust_list]

    if mode == "train":
        # split data into training and validation data
        velocity_train, velocity_val = train_test_split(velocity_list, test_size=validation_data_fraction,
                                                        train_size=1 - validation_data_fraction, shuffle=False)

        torque_train, torque_val = train_test_split(torque_list, test_size=validation_data_fraction,
                                                    train_size=1 - validation_data_fraction, shuffle=False)

        thrust_train, thrust_val = train_test_split(thrust_list, test_size=validation_data_fraction,
                                                    train_size=1 - validation_data_fraction, shuffle=False)

        # load training and validation data
        train_loader_velocity = data_loader(velocity_train, device, shuffle=False)
        val_loader_velocity = data_loader(velocity_val, device, shuffle=False)

        train_loader_torque = data_loader(torque_train, device, shuffle=False)
        val_loader_torque = data_loader(torque_val, device, shuffle=False)

        train_loader_thrust = data_loader(thrust_train, device, shuffle=False)
        val_loader_thrust = data_loader(thrust_val, device, shuffle=False)

        trainer = PredictionTrainer(encoder_thrust, encoder_torque, decoder, epochs,
                                    train_loader_thrust, train_loader_torque, train_loader_velocity,
                                    val_loader_thrust, val_loader_torque, val_loader_velocity,
                                    device, optimizer)

        # do training and get losses
        average_training_losses, average_validation_losses = trainer.train_model()

        # save weights
        torch.save(decoder.state_dict(), weights_path_decoder)

        # record average training and validation losses per epoch
        write_losses(average_training_losses, average_validation_losses)

        # visualise average training and validation losses per epoch
        if visualise_training_and_validation_loss:
            losses_plot(average_training_losses, average_validation_losses, plot_loss_50_epoch_skip)

    # load all data
    val_loader_velocity = data_loader(velocity_list, device, shuffle=False)
    val_loader_torque = data_loader(torque_list, device, shuffle=False)
    val_loader_thrust = data_loader(thrust_list, device, shuffle=False)

    # load model weights and activate evaluation if training is off
    if mode != "train":
        decoder.load_state_dict(torch.load(weights_path_decoder))
        decoder.eval()

    # visualise reconstruction if visualisation is on
    if visualise_scatter:
        prediction_reconstruction_scatter_plot(encoder_thrust, encoder_torque, decoder, device, velocity_list,
                                               "velocity", val_loader_thrust, val_loader_torque,val_loader_velocity,
                                               show_y_equals_x)

    plt.show()
