import matplotlib.pyplot as plt
import numpy as np
import torch
from ReaderWriter import read_data_lists, read_losses
from PredictionTrainer import encode_inputs


def read_and_plot_data(normalization=False):
    """

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


def read_and_plot_losses(plot_50_epoch_skip=False):
    """

    """
    # read losses
    train_loss_list, validation_loss_list = read_losses()

    # plot losses
    losses_plot(train_loss_list, validation_loss_list, plot_50_epoch_skip)
    plt.show()


def data_list_plot(list, name, start_index=0, end_index=None, data_fraction=None):
    """

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


def reconstruction_scatter_plot(vae, data, val_loader, data_type):
    """

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

    reconstruction_scatter_plot_helper(originals, reconstructions, data, data_type)


def prediction_reconstruction_scatter_plot(encoder1, encoder2, decoder, device, data_to_predict, data_to_predict_type,
                                           val_loader_enc1, val_loader_enc2):
    """

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

    reconstruction_scatter_plot_helper(originals, reconstructions, data_to_predict, data_to_predict_type)


def reconstruction_scatter_plot_helper(originals, reconstructions, data, data_type):
    """

    """
    # make scatter plot of originals and reconstructions
    plt.figure()
    plt.scatter(originals, reconstructions)

    # make plot of y = x
    min_data = min(data)
    max_data = max(data)
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

    """
    originals_array = np.array(originals)
    reconstructions_array = np.array(reconstructions)

    difference_array = reconstructions_array - originals_array

    signed_error_mean = np.mean(difference_array)
    signed_error_std = np.std(difference_array)

    print("Signed Error Mean: ", signed_error_mean)
    print("Signed Error STD: ", signed_error_std)


if __name__ == "__main__":
    read_and_plot_data(True)
