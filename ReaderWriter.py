import csv


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


def read_losses():
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

        # return losses
        return train_loss_list, validation_loss_list


def write_losses(average_training_losses, average_validation_losses):
    with open('loss_record.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        headers = ["avg_train_loss", "avg_val_loss"]
        csv_writer.writerow(headers)

        for avg_train_loss, avg_val_loss in zip(average_training_losses, average_validation_losses):
            row = [avg_train_loss, avg_val_loss]
            csv_writer.writerow(row)
