"""
Contains functions for the production of plots, used to visualise
data points, training losses, reconstructions and predictions.
Also contains a function for the printing of the mean and
standard deviation of a data set.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from ReaderWriter import read_data_lists
from PredictionTrainer import encode_inputs


def read_and_plot_data(normalization=False):
    """
    Reads velocity, thrust and torque data from Data.txt and plots them.

    Args:
        normalization (bool, optional): brings all data to the same order of magnitude, O(10),
        before plotting (default: False).
    """
    # read data
    velocity_list, thrust_list, torque_list = read_data_lists()

    if normalization:
        # multiply velocities by 10
        velocity_list = [x * 10 for x in velocity_list]

        # divide thrusts by 10
        thrust_list = [x / 10 for x in thrust_list]

    # plot data
    data_list_plot(thrust_list, 'Thrust')
    data_list_plot(torque_list, 'Torque')
    data_list_plot(velocity_list, 'Velocity_x')

    plt.show()


def data_list_plot(list, name, start_index=0, end_index=None, data_fraction=None):
    """
    Plots data list or part of it.

    Args:
        list (list): data list.
        name (string): name of data.
        start_index (int, optional): index of data to start the plot from (default: 0).
        end_index (int, optional): index of data to end the plot on (default: None).
        data_fraction (float, optional): fraction of data to plot (default: None).
    """
    # set default end index(exclusive)
    if end_index is None:
        end_index = len(list)

    # set non-default end index(exclusive) using the given fraction of data to plot
    if data_fraction is not None:
        start_index = 0
        end_index = round(len(list) * data_fraction)

    # get number of data points
    data_point_number = end_index - start_index

    # plot data
    plt.figure()
    plt.plot(np.linspace(start_index, end_index - 1, data_point_number), list[start_index:end_index])
    plt.ylabel(name)

    # plot data histogram
    plt.figure()
    plt.hist(list[start_index:end_index])
    plt.title(name)

    # print data stats
    data_arr = np.array(list[start_index:end_index])

    signed_error_mean = np.mean(data_arr)
    signed_error_std = np.std(data_arr)

    print(name, "mean: ", signed_error_mean)
    print(name, "std: ", signed_error_std)


def losses_plot(average_training_losses, average_validation_losses, plot_1_epoch_skip=True,
                plot_50_epoch_skip=False, title="Training Loss"):
    """
    Plots losses.

    Args:
        average_training_losses (list): average training losses.
        average_validation_losses (list): average validation losses.
        plot_1_epoch_skip (bool, optional): if set to True makes an extra losses plot
        skipping the first epoch(default: True).
        plot_50_epoch_skip (bool, optional): if set to True makes an extra losses plot
        skipping the first 50 epochs (default: False).
        title (string, optional): title of plot (default: "Training Loss").
    """
    # get number of epochs
    epochs = len(average_training_losses)

    # plot losses
    plt.figure()
    epochs_arr = np.linspace(0, epochs - 1, epochs)
    plt.plot(epochs_arr, average_training_losses, label="Avg Train Loss", color="blue")
    plt.plot(epochs_arr, average_validation_losses, label="Avg Val Loss", color="red")

    # set axis labels and legend
    plt.ylabel("Avg Loss")
    plt.xlabel("Epoch")
    plt.legend(loc='best')
    plt.title(title)

    if plot_1_epoch_skip:
        # plot losses skipping 1st epoch
        plt.figure()
        epochs_arr = np.linspace(1, epochs - 1, epochs - 1)
        plt.plot(epochs_arr, average_training_losses[1:], label="Avg Train Loss", color="blue")
        plt.plot(epochs_arr, average_validation_losses[1:], label="Avg Val Loss", color="red")

        # set axis labels and legend
        plt.ylabel("Avg Loss")
        plt.xlabel("Epoch")
        plt.legend(loc='best')
        plt.title(title)

    if plot_50_epoch_skip:
        # plot losses skipping first 50 epochs
        plt.figure()
        epochs_arr = np.linspace(50, epochs - 1, epochs - 50)
        plt.plot(epochs_arr, average_training_losses[50:], label="Avg Train Loss", color="blue")
        plt.plot(epochs_arr, average_validation_losses[50:], label="Avg Val Loss", color="red")

        # set axis labels and legend
        plt.ylabel("Avg Loss")
        plt.xlabel("Epoch")
        plt.legend(loc='best')
        plt.title(title)


def reconstruction_scatter_plot(vae, val_loader, data_type):
    """
    Passes the values in val_loader through VAE and
    makes a scatter plot, having the original value
    on the x axis and the reconstructed value on the y axis.

    Args:
        vae (VAE): variational autoencoder.
        val_loader (DataLoader): data loader with data.
        data_type (string): type of data.
    """
    # get lists of original data and reconstructions
    reconstructions = []
    originals = []

    for inputs in val_loader:
        # get inputs
        inputs = inputs[0]

        # get outputs
        outputs = vae(inputs)

        # reformat and flatten outputs
        outputs = outputs[0].detach().view(-1).to(torch.device('cpu'))
        outputs = outputs.numpy()

        # add outputs/reconstructions to list
        reconstructions.extend(outputs)

        # reformat and flatten inputs
        inputs = inputs.view(-1).to(torch.device('cpu'))
        inputs = inputs.numpy()

        # add inputs/originals to list
        originals.extend(inputs)

    reconstruction_scatter_plot_helper(originals, reconstructions, data_type)


def prediction_reconstruction_scatter_plot(encoder1, encoder2, decoder, device, data_to_predict, data_to_predict_type,
                                           val_loader_enc1, val_loader_enc2):
    """
    Passes the values in val_loader_enc1 through encoder1 and
    those in val_loader_enc2 through encoder2. Takes the resulting
    means and log variances and passes them through decoder to
    make a prediction. Makes a scatter plot, having the original value
    on the x axis and the predicted value on the y axis.

    Args:
        encoder1 (Encoder): encoder for the data in val_loader_enc1.
        encoder2 (Encoder): encoder for the data in val_loader_enc2.
        decoder (Decoder): decoder for the encoded data.
        device (torch.device): torch device.
        data_to_predict (list): list of data to be predicted.
        data_to_predict_type (string): type of data to be predicted.
        val_loader_enc1 (DataLoader): data loader with data for encoder1.
        val_loader_enc2 (DataLoader): data loader with data for encoder2.
    """
    # get lists of original data and reconstructions
    reconstructions = []
    originals = data_to_predict

    for inputs1, inputs2 in zip(val_loader_enc1, val_loader_enc2):
        # get encoded inputs
        encoded_inputs_tensor = encode_inputs(inputs1, inputs2, encoder1, encoder2, device)

        # get outputs
        outputs = decoder(encoded_inputs_tensor)

        # reformat and flatten outputs
        outputs = outputs.detach().view(-1).to(torch.device('cpu'))
        outputs = outputs.numpy()

        # add outputs/reconstructions to list
        reconstructions.extend(outputs)

    reconstruction_scatter_plot_helper(originals, reconstructions, data_to_predict_type)


def reconstruction_scatter_plot_helper(originals, reconstructions, data_type):
    """
    Helper function for reconstruction_scatter_plot and prediction_reconstruction_scatter_plot.
    Makes a scatter plot, having the original value
    on the x axis and the reconstructed/predicted value
    on the y axis. Prints the signed error mean &
    standard deviation after the plotting is finished.

    Args:
        originals (list): original data.
        reconstructions (list): reconstructed data.
        data_type (string): type of data.
    """
    # make scatter plot of originals and reconstructions
    plt.figure()
    plt.scatter(originals, reconstructions)

    # make plot of y = x
    min_data = min(originals)
    max_data = max(originals)
    straight_line_data = np.linspace(min_data, max_data)
    plt.plot(straight_line_data, straight_line_data, color="black")

    # set axis labels and title
    plt.ylabel("reconstruction")
    plt.xlabel("original")
    plt.title(data_type)

    # print signed error mean & standard deviation
    print_signed_error_mean_and_std(originals, reconstructions)


def print_signed_error_mean_and_std(originals, reconstructions):
    """
    Prints the signed error mean & standard deviation of the given data.

    Args:
        originals (list): original data.
        reconstructions (list): reconstructed data.
    """
    originals_array = np.array(originals)
    reconstructions_array = np.array(reconstructions)

    difference_array = reconstructions_array - originals_array

    signed_error_mean = np.mean(difference_array)
    signed_error_std = np.std(difference_array)

    print("Signed Error Mean: ", signed_error_mean)
    print("Signed Error STD: ", signed_error_std)

