"""
Contains functions for the reading of velocity, thrust and torque data and losses from files, and
the writing of losses to files.
"""

import csv


def read_data_lists(file_name='Data.txt'):
    """
    Reads velocity, thrust and torque data from a file and returns them as lists.

    Args:
        file_name (string, optional): the name of the file from which the velocity, thrust and torque data
        is to be read (default: "Data.txt").

    Returns:
            tuple: (list: velocity data, list: thrust data, list: torque data).
    """
    velocity_list = []
    thrust_list = []
    torque_list = []

    # open Data.txt file
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        first_line = True

        for row in csv_reader:
            if first_line:
                # skip first line with headers
                first_line = False
            else:
                # add data points to appropriate lists
                velocity_list.append(row[0])
                thrust_list.append(row[1])
                torque_list.append(row[2])

        # convert strings to floats
        velocity_list = [float(i) for i in velocity_list]
        thrust_list = [float(i) for i in thrust_list]
        torque_list = [float(i) for i in torque_list]

    return velocity_list, thrust_list, torque_list


def read_losses(file_name='loss_record.csv'):
    """
    Reads velocity, thrust and torque losses from a file and returns them as lists.

    Args:
        file_name (string, optional): the name of the file from which the velocity, thrust and torque losses
        are to be read (default: "loss_record.csv").

    Returns:
            tuple: (list: training losses, list: validation losses).
    """
    train_loss_list = []
    validation_loss_list = []

    # open loss_record.csv file
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        first_line = True
        even_line = False

        for row in csv_reader:
            if first_line:
                # skip first line with headers
                first_line = False
                # toggle even line flag
                even_line = True
            else:
                if not even_line:
                    # add data points to appropriate lists
                    train_loss_list.append(row[0])
                    validation_loss_list.append(row[1])
                    even_line = True
                else:
                    # skip empty line
                    even_line = False

        # convert strings to floats
        train_loss_list = [float(i) for i in train_loss_list]
        validation_loss_list = [float(i) for i in validation_loss_list]

        return train_loss_list, validation_loss_list


def write_general_losses(average_training_losses, average_validation_losses, file_name):
    """
    Write average training and validation losses to a file.

    Args:
        average_training_losses (list): list of average training losses.
        average_validation_losses (list): list of average validation losses.
        file_name (string): name of file where losses are recorded.
    """
    # open/create loss_record.csv file
    with open(file_name, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')

        # write headers
        headers = ["avg_train_loss", "avg_val_loss"]
        csv_writer.writerow(headers)

        # write data
        for avg_train_loss, avg_val_loss in zip(average_training_losses, average_validation_losses):
            row = [avg_train_loss, avg_val_loss]
            csv_writer.writerow(row)


def write_losses(average_total_training_losses, average_total_validation_losses,
                 average_mse_training_losses, average_mse_validation_losses):
    """
    Write average training and validation losses and mean squared errors to files:
        losses: loss_record.csv
        mean squared errors: mse_loss_record.csv

    Args:
        average_total_training_losses (list): list of average training losses.
        average_total_validation_losses (list): list of average validation losses.
        average_mse_training_losses (list): list of average training mean squared errors.
        average_mse_validation_losses (list): list of average validation mean squared errors.
    """
    write_general_losses(average_total_training_losses, average_total_validation_losses, 'loss_record.csv')
    write_general_losses(average_mse_training_losses, average_mse_validation_losses, 'mse_loss_record.csv')


