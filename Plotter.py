import matplotlib.pyplot as plt
import numpy as np
import torch
from ReaderWriter import read_data_lists, read_losses
from PredictionTrainer import encode_inputs


def read_and_plot_data():
    velocity_list, thrust_list, torque_list = read_data_lists()

    data_list_plot(velocity_list, 'Velocity_x')
    data_list_plot(thrust_list, 'Thrust')
    data_list_plot(torque_list, 'Torque')

    plt.show()


def read_and_plot_losses(plot_50_epoch_skip=False):
    # read losses
    train_loss_list, validation_loss_list = read_losses()

    # plot losses
    losses_plot(train_loss_list, validation_loss_list, plot_50_epoch_skip)
    plt.show()


def data_list_plot(list, name, start_index=0, end_index=None, data_fraction=None):
    if end_index is None:
        end_index = len(list)

    if data_fraction is not None:
        start_index = 0
        end_index = round(len(list) * data_fraction)

    size = end_index - start_index

    plt.figure()
    plt.plot(np.linspace(start_index, end_index - 1, size), list[start_index:end_index])
    plt.ylabel(name)


def losses_plot(average_training_losses, average_validation_losses, plot_50_epoch_skip=False):
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
    plt.title("Training Loss")

    # plot losses skipping 1st epoch
    plt.figure()
    epochs_arr = np.linspace(1, epochs - 1, epochs - 1)
    plt.plot(epochs_arr, average_training_losses[1:], label="Avg Train Loss", color="blue")
    plt.plot(epochs_arr, average_validation_losses[1:], label="Avg Val Loss", color="red")

    # set axis labels and legend
    plt.ylabel("Avg Loss")
    plt.xlabel("Epoch")
    plt.legend(loc='best')
    plt.title("Training Loss")

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
        plt.title("Training Loss")


def reconstruction_scatter_plot(vae, data, val_loader, show_y_equals_x, data_type, drop_outliers):
    # get lists of original data and reconstructions
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

    max_data = reconstruction_scatter_plot_helper(originals, reconstructions, data, data_type, show_y_equals_x)

    # set up drop of outliers in visualisation if turned on
    unit = max_data / 21
    if drop_outliers:
        y_upper_limit = max_data + unit
    else:
        y_upper_limit = None
    plt.ylim(top=y_upper_limit, bottom=-unit)


def prediction_reconstruction_scatter_plot(encoder1, encoder2, decoder, device, data_to_predict, data_to_predict_type,
                                           val_loader_enc1, val_loader_enc2, val_loader_pred, show_y_equals_x, sampling):
    # get lists of original data and reconstructions
    reconstructions = []
    originals = data_to_predict

    for inputs1, inputs2, targets in zip(val_loader_enc1, val_loader_enc2, val_loader_pred):
        encoded_inputs_tensor = encode_inputs(inputs1, inputs2, encoder1, encoder2, device, sampling)
        targets = targets[0]

        outputs = decoder(encoded_inputs_tensor)
        outputs = outputs.detach().view(-1).to(torch.device('cpu'))
        outputs = outputs.numpy()
        reconstructions.extend(outputs)

    reconstruction_scatter_plot_helper(originals, reconstructions, data_to_predict, data_to_predict_type,
                                       show_y_equals_x)


def reconstruction_scatter_plot_helper(originals, reconstructions, data, data_type, show_y_equals_x):
    # make scatter plot of originals and reconstructions
    plt.figure()
    plt.scatter(originals, reconstructions)

    # make plot of y = x if turned on
    min_data = min(data)
    max_data = max(data)
    if show_y_equals_x:
        straight_line_data = np.linspace(min_data, max_data)
        plt.plot(straight_line_data, straight_line_data, color="black")

    # set axis labels and title
    plt.ylabel("reconstruction")
    plt.xlabel("original")
    plt.title(data_type)

    return max_data


if __name__ == "__main__":
    read_and_plot_losses(False)
