import sys
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


data_to_predict_type = "thrust"
mode = "train"
epochs = 10
plot_loss_50_epoch_skip = False
sampling = False

convolution_channel_size_1 = 16
convolution_channel_size_2 = 8
convolution_channel_size_3 = 16
convolution_channel_size_4 = 4
fully_connected_unit_size = 400
convolution_kernel = 3

weights_path_thrust = "./vae_net_thrust.pth"
weights_path_torque = "./vae_net_torque.pth"
weights_path_velocity = "./vae_net_velocity.pth"
weights_path_decoder = "./vae_net_prediction_decoder.pth"
lr = 1e-4


class PredictionDecoder(nn.Module):

    def __init__(self):
        super(PredictionDecoder, self).__init__()

        # latent space transformation
        if sampling:
            self.fc_lat = nn.Linear(2 * latent_dimensions, fully_connected_unit_size)
        else:
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
    # get data
    velocity_list, thrust_list, torque_list = read_data_lists()
    velocity_list = [x * 10 for x in velocity_list]
    thrust_list = [x / 10 for x in thrust_list]

    # choose data to be predicted
    if data_to_predict_type == "velocity":
        data_enc1 = thrust_list
        data_enc2 = torque_list
        data_pred = velocity_list

        weights_path_enc1 = weights_path_thrust
        weights_path_enc2 = weights_path_torque
    elif data_to_predict_type == "torque":
        data_enc1 = thrust_list
        data_enc2 = velocity_list
        data_pred = torque_list

        weights_path_enc1 = weights_path_thrust
        weights_path_enc2 = weights_path_velocity
    elif data_to_predict_type == "thrust":
        data_enc1 = torque_list
        data_enc2 = velocity_list
        data_pred = thrust_list

        weights_path_enc1 = weights_path_torque
        weights_path_enc2 = weights_path_velocity
    else:
        sys.exit("Incorrect data type.")

    # set seed
    torch.manual_seed(seed)

    # setup neural networks and optimizer
    vae_enc1 = VAE(latent_dimensions)
    vae_enc2 = VAE(latent_dimensions)
    decoder = PredictionDecoder()
    optimizer = Adam(decoder.parameters(), lr=lr)

    # load model weights
    vae_enc1.load_state_dict(torch.load(weights_path_enc1))
    vae_enc2.load_state_dict(torch.load(weights_path_enc2))

    # get encoders
    encoder1 = vae_enc1.encoder
    encoder2 = vae_enc2.encoder

    # set encoders to evaluation
    encoder1.eval()
    encoder2.eval()

    # choose device(cpu or gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoder1.to(device)
    encoder2.to(device)
    decoder.to(device)

    if mode == "train":
        # split data into training and validation data
        pred_train, pred_val = train_test_split(data_pred, test_size=validation_data_fraction,
                                                train_size=1 - validation_data_fraction, shuffle=False)

        enc2_train, enc2_val = train_test_split(data_enc2, test_size=validation_data_fraction,
                                                train_size=1 - validation_data_fraction, shuffle=False)

        enc1_train, enc1_val = train_test_split(data_enc1, test_size=validation_data_fraction,
                                                train_size=1 - validation_data_fraction, shuffle=False)

        # load training and validation data
        train_loader_pred = data_loader(pred_train, device, shuffle=False)
        val_loader_pred = data_loader(pred_val, device, shuffle=False)

        train_loader_enc2 = data_loader(enc2_train, device, shuffle=False)
        val_loader_enc2 = data_loader(enc2_val, device, shuffle=False)

        train_loader_enc1 = data_loader(enc1_train, device, shuffle=False)
        val_loader_enc1 = data_loader(enc1_val, device, shuffle=False)

        trainer = PredictionTrainer(encoder1, encoder2, decoder, epochs,
                                    train_loader_enc1, train_loader_enc2, train_loader_pred,
                                    val_loader_enc1, val_loader_enc2, val_loader_pred,
                                    device, optimizer, sampling)

        # do training and get losses
        average_training_losses, average_validation_losses = trainer.train_model()

        # save weights
        torch.save(decoder.state_dict(), weights_path_decoder)

        # record average training and validation losses per epoch
        write_losses(average_training_losses, average_validation_losses)

        # visualise average training and validation losses per epoch
        losses_plot(average_training_losses, average_validation_losses, plot_loss_50_epoch_skip)

    # load all data
    val_loader_pred = data_loader(data_pred, device, shuffle=False)
    val_loader_enc2 = data_loader(data_enc2, device, shuffle=False)
    val_loader_enc1 = data_loader(data_enc1, device, shuffle=False)

    # load model weights and activate evaluation if training is off
    if mode != "train":
        decoder.load_state_dict(torch.load(weights_path_decoder))
        decoder.eval()

    # visualise reconstruction
    prediction_reconstruction_scatter_plot(encoder1, encoder2, decoder, device, data_pred,
                                           data_to_predict_type, val_loader_enc1, val_loader_enc2,
                                           sampling)

    plt.show()
