"""
Contains the prediction decoder class, PredictionDecoder, and helper functions.

When the file is run, the behaviour is governed by the changeable parameters:
    - data_to_predict_type (string): sets the data type to be predicted by the program from among
        "velocity", "thrust" and "torque".
    - mode (string): decides the mode of action of the program:
        - mode = "train", the prediction decoder is trained with the chosen data type,
        saves its weights to a file with a name given by the weights_path_decoder variable, saves the training and
        validation losses to loss_record.csv and produces plots of the losses.
        - mode != "train"(or if mode = "train" and after the training is done), the prediction decoder uses
        pretrained weights from a file with a name given by the weights_path_decoder variable to predict
        the chosen data type from the other two and shows a scatter plot, having the original value on the x axis
        and the reconstructed value on the y axis.
    - epochs (int): decides the number of epochs the prediction decoder trains for if it
        is in train mode.
    - plot_loss_1_epoch_skip (bool): decides whether to produce an additional plot of the
        training and validation losses skipping the first epoch.
        It should only be used when the chosen number of epochs is 2 or more.
    - plot_loss_50_epoch_skip (bool): decides whether to produce an additional plot of the
        training and validation losses skipping the first 50 epochs.
        It should only be used when the chosen number of epochs is 50 or more.
    - validation_data_fraction (float): the fraction of the data which is to be used for validation.

Based on:
    File Name: main.py
    Developed by: Nikolas Pitsillos, PhD Candidate in Computer Vision and Autonomous Systems @ UofG
    Taken from: https://github.com/npitsillos/mnist-vae/blob/master/main.py
    Described at: https://npitsillos.github.io/posts/2020/05/mnistvae/
"""

import sys
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from Plotter import losses_plot, prediction_reconstruction_scatter_plot
from PredictionDecoderVelocity import PredictionDecoderVelocity
from PredictionTrainer import PredictionTrainer
from ReaderWriter import read_data_lists, write_general_losses
from VAE import data_loader, VAE, latent_dimensions, data_sequence_size, seed, lr, Decoder, \
    convolution_channel_size_4, fully_connected_unit_size

# variables that can be changed
data_to_predict_type = "velocity"
mode = "test"
epochs = 3
plot_loss_1_epoch_skip = True
plot_loss_50_epoch_skip = False
validation_data_fraction = 0.2

# variables that should not be changed
weights_path_thrust = "./vae_net_thrust.pth"
weights_path_torque = "./vae_net_torque.pth"
weights_path_velocity = "./vae_net_velocity.pth"
weights_path_decoder = "./vae_net_prediction_decoder_%s.pth" % data_to_predict_type


class PredictionDecoder(Decoder):
    """
    Decoder used for predicting one parameter from the encoded means and log variances
    of the other two parameters.
    """

    def __init__(self):
        """
        Initializes internal PredictionDecoder state.
        """
        super(PredictionDecoder, self).__init__(latent_dimensions, convolution_channel_size_4 * data_sequence_size)

        # latent space transformation
        self.fc_lat = nn.Linear(4 * latent_dimensions, fully_connected_unit_size)


def get_data_and_weights():
    """
    Reads all data types and the weights for encoder1 and encoder 2,
    and returns them.

    Returns:
            tuple: (list: data for encoder1, list: data for encoder2, list: data to be predicted,
                string: weights path for vae containing encoder1,
                string: weights path for vae containing encoder2).
    """
    # get all data
    velocity_list, thrust_list, torque_list = read_data_lists()

    # multiply velocities by 10
    velocity_list = [x * 10 for x in velocity_list]

    # divide thrusts by 10
    thrust_list = [x / 10 for x in thrust_list]

    # choose correct data and weights
    if data_to_predict_type == "velocity":
        data_to_encode1 = thrust_list
        data_to_encode2 = torque_list
        data_to_predict = velocity_list

        weights_path_vae1 = weights_path_thrust
        weights_path_vae2 = weights_path_torque
    elif data_to_predict_type == "torque":
        data_to_encode1 = thrust_list
        data_to_encode2 = velocity_list
        data_to_predict = torque_list

        weights_path_vae1 = weights_path_thrust
        weights_path_vae2 = weights_path_velocity
    elif data_to_predict_type == "thrust":
        data_to_encode1 = torque_list
        data_to_encode2 = velocity_list
        data_to_predict = thrust_list

        weights_path_vae1 = weights_path_torque
        weights_path_vae2 = weights_path_velocity
    else:
        sys.exit("Incorrect data type.")

    return data_to_encode1, data_to_encode2, data_to_predict, weights_path_vae1, weights_path_vae2


def setup(data_to_encode1, data_to_encode2, data_to_predict, weights_path_vae1, weights_path_vae2):
    """
    Sets up the seed, the encoders, the decoder, the optimizer, the torch device,
    the splitting of the data into training and validation data sets,
    the loading of the training and validation data into data loaders, and the trainer.

    Args:
        data (list): list of data points(floats).

    Returns:
            tuple: (torch.device: torch device, Encoder: encoder1, Encoder: encoder2,
                PredictionDecoder/PredictionDecoderVelocity: decoder, PredictionTrainer: trainer).
    """
    # set seed
    torch.manual_seed(seed)

    # setup neural networks and optimizer
    vae1 = VAE(latent_dimensions)
    vae2 = VAE(latent_dimensions)
    if data_to_predict_type == "velocity":
        decoder = PredictionDecoderVelocity()
    else:
        decoder = PredictionDecoder()
    optimizer = Adam(decoder.parameters(), lr=lr)

    # load model weights
    vae1.load_state_dict(torch.load(weights_path_vae1))
    vae2.load_state_dict(torch.load(weights_path_vae2))

    # get encoders
    encoder1 = vae1.encoder
    encoder2 = vae2.encoder

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
        training_data_to_encode1, validation_data_to_encode1 = train_test_split(data_to_encode1,
                                                                                test_size=validation_data_fraction,
                                                                                train_size=1 - validation_data_fraction,
                                                                                shuffle=False)
        training_data_to_encode2, validation_data_to_encode2 = train_test_split(data_to_encode2,
                                                                                test_size=validation_data_fraction,
                                                                                train_size=1 - validation_data_fraction,
                                                                                shuffle=False)
        training_data_to_predict, validation_data_to_predict = train_test_split(data_to_predict,
                                                                                test_size=validation_data_fraction,
                                                                                train_size=1 - validation_data_fraction,
                                                                                shuffle=False)

        # load training and validation data
        training_loader_to_encode1 = data_loader(training_data_to_encode1, device, shuffle=False)
        validation_loader_to_encode1 = data_loader(validation_data_to_encode1, device, shuffle=False)

        training_loader_to_encode2 = data_loader(training_data_to_encode2, device, shuffle=False)
        validation_loader_to_encode2 = data_loader(validation_data_to_encode2, device, shuffle=False)

        training_loader_to_predict = data_loader(training_data_to_predict, device, shuffle=False)
        validation_loader_to_predict = data_loader(validation_data_to_predict, device, shuffle=False)

        trainer = PredictionTrainer(encoder1, encoder2, decoder, epochs,
                                    training_loader_to_encode1, training_loader_to_encode2, training_loader_to_predict,
                                    validation_loader_to_encode1, validation_loader_to_encode2,
                                    validation_loader_to_predict, device, optimizer)
    else:
        trainer = None

    return device, encoder1, encoder2, decoder, trainer


if __name__ == "__main__":
    # get data and weights
    data_to_encode1, data_to_encode2, data_to_predict, weights_path_vae1, weights_path_vae2 = get_data_and_weights()

    # setup
    device, encoder1, encoder2, decoder, trainer = setup(data_to_encode1, data_to_encode2, data_to_predict,
                                                         weights_path_vae1, weights_path_vae2)

    # train model if training is on
    if mode == "train":
        # do training and get losses
        average_training_losses, average_validation_losses = trainer.train_model()

        # save weights
        torch.save(decoder.state_dict(), weights_path_decoder)

        # record average training and validation losses per epoch
        write_general_losses(average_training_losses, average_validation_losses, 'loss_record.csv')

        # visualise average training and validation losses per epoch
        losses_plot(average_training_losses, average_validation_losses,
                    plot_1_epoch_skip=plot_loss_1_epoch_skip, plot_50_epoch_skip=plot_loss_50_epoch_skip)

    # load all data
    validation_loader_to_encode1 = data_loader(data_to_encode1, device, shuffle=False)
    validation_loader_to_encode2 = data_loader(data_to_encode2, device, shuffle=False)
    validation_loader_to_predict = data_loader(data_to_predict, device, shuffle=False)

    # load model weights and activate evaluation if training is off
    if mode != "train":
        decoder.load_state_dict(torch.load(weights_path_decoder))
        decoder.eval()

    # visualise reconstruction
    prediction_reconstruction_scatter_plot(encoder1, encoder2, decoder, device, data_to_predict,
                                           data_to_predict_type, validation_loader_to_encode1,
                                           validation_loader_to_encode2)

    plt.show()
