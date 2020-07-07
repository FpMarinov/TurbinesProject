import matplotlib.pyplot as plt
import numpy as np
import csv

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_lists():
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


def plot_list(list, name, start_index=0, end_index=None, data_fraction=None):

    if end_index is None:
        end_index = len(list)

    if data_fraction is not None:
        start_index = 0
        end_index = round(len(list) * data_fraction)

    size = end_index - start_index

    plt.figure()
    plt.plot(np.linspace(start_index, end_index - 1, size), list[start_index:end_index])
    plt.ylabel(name)


if __name__ == "__main__":

    velocity_list, thrust_list, torque_list = get_lists()

    plot_list(velocity_list, 'Velocity_x')
    plot_list(thrust_list, 'Thrust')
    plot_list(torque_list, 'Torque')

    plt.show()

    data_train, data_test = train_test_split(velocity_list, test_size=0.2, train_size=0.8, shuffle=False)
    print(data_test)





