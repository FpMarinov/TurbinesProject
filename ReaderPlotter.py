import matplotlib.pyplot as plt
import numpy as np
import csv

import torch


def read_data_lists():
    velocity_list = []
    thrust_list = []
    torque_list = []

    with open('Data.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        first_line = True

        for row in csv_reader:
            if first_line:
                first_line = False
            else:
                velocity_list.append(row[0])
                thrust_list.append(row[1])
                torque_list.append(row[2])

        # convert strings to numbers
        velocity_list = [float(i) for i in velocity_list]
        thrust_list = [float(i) for i in thrust_list]
        torque_list = [float(i) for i in torque_list]

    return velocity_list, thrust_list, torque_list


def read_and_plot_data():
    velocity_list, thrust_list, torque_list = read_data_lists()

    plot_data_list(velocity_list, 'Velocity_x')
    plot_data_list(thrust_list, 'Thrust')
    plot_data_list(torque_list, 'Torque')

    plt.show()


def read_and_plot_losses():
    train_loss_list = []
    validation_loss_list = []

    with open('loss_record.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        first_line = True
        even_line = False

        for row in csv_reader:
            if first_line:
                first_line = False
                even_line = True
            else:
                if not even_line:
                    train_loss_list.append(row[0])
                    validation_loss_list.append(row[1])
                    even_line = True
                else:
                    even_line = False

        # convert strings to numbers
        train_loss_list = [float(i) for i in train_loss_list]
        validation_loss_list = [float(i) for i in validation_loss_list]

        # plot losses
        plot_losses(train_loss_list, validation_loss_list)
        plt.show()


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

    # make scatter plot of originals and reconstructions
    plt.figure()
    plt.scatter(originals, reconstructions)

    # make plot of y = x if turned on
    min_data = min(data)
    max_data = max(data)
    if show_y_equals_x:
        straight_line_data = np.linspace(min(data), max(data))
        plt.plot(straight_line_data, straight_line_data, color="black")

    # set axis labels and title
    plt.ylabel("reconstruction")
    plt.xlabel("original")
    plt.title(data_type)

    # set up drop of outliers in visualisation if turned on
    unit = max_data / 21
    if drop_outliers:
        y_upper_limit = max_data + unit
    else:
        y_upper_limit = None
    plt.ylim(top=y_upper_limit, bottom=-unit)


def plot_data_list(list, name, start_index=0, end_index=None, data_fraction=None):
    if end_index is None:
        end_index = len(list)

    if data_fraction is not None:
        start_index = 0
        end_index = round(len(list) * data_fraction)

    size = end_index - start_index

    plt.figure()
    plt.plot(np.linspace(start_index, end_index - 1, size), list[start_index:end_index])
    plt.ylabel(name)


def plot_losses(average_training_losses, average_validation_losses):
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


if __name__ == "__main__":
    read_and_plot_losses()
