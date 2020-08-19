import csv


def read_data_lists(file_name='Data.txt'):
    """

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

    """
    write_general_losses(average_total_training_losses, average_total_validation_losses, 'loss_record.csv')
    write_general_losses(average_mse_training_losses, average_mse_validation_losses, 'mse_loss_record.csv')


if __name__ == "__main__":
    velocity_list, thrust_list, torque_list = read_data_lists('Data.txt')
    print('1D&3D: ', len(velocity_list))

