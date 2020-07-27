from sklearn.model_selection import train_test_split
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from trainer import Trainer
from ReaderPlotter import read_data_lists
from ReaderPlotter import losses_plot
from ReaderPlotter import reconstruction_scatter_plot
from ReaderPlotter import read_losses
from VAE import loss_fn
from VAE import VAE
import matplotlib.pyplot as plt
import csv

data_type = "velocity"
extra_epochs = 100
visualise_scatter = True
drop_scatter_outliers = False
show_y_equals_x = True
visualise_training_and_validation_loss = True
drop_infinity_from_loss_record_calc = False
plot_loss_50_epoch_skip = True

data_sequence_size = 5
batch_size = 5
convolution_channel_size_1 = 16
convolution_channel_size_2 = 8
convolution_channel_size_3 = 16
convolution_channel_size_4 = 4
fully_connected_unit_size = 400
latent_dimensions = 1
convolution_kernel = 3

weights_path = "./vae_net_%s.pth" % data_type

seed = 1
lr = 1e-4
validation_data_fraction = 0.2
print_freq = 10


if __name__ == "__main__":
    # get data
    velocity_list, thrust_list, torque_list = read_data_lists()
    if data_type == "velocity":
        data = velocity_list
        data = [x * 10 for x in data]
    elif data_type == "thrust":
        data = thrust_list
        data = [x / 10 for x in data]
    elif data_type == "torque":
        data = torque_list
    else:
        sys.exit("Incorrect data_type.")

    # set seed
    torch.manual_seed(seed)

    # setup neural network and optimizer
    vae = VAE(latent_dimensions)
    optimizer = Adam(vae.parameters(), lr=lr)

    # choose device(cpu or gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vae.to(device)

    # split data into training and validation data
    data_train, data_val = train_test_split(data, test_size=validation_data_fraction,
                                            train_size=1-validation_data_fraction, shuffle=False)

    # format training data
    tensor_train = torch.FloatTensor(data_train).view(-1, 1, data_sequence_size)
    tensor_train = tensor_train.to(device)
    train_dataset = TensorDataset(tensor_train)

    # format validation data
    tensor_val = torch.FloatTensor(data_val).view(-1, 1, data_sequence_size)
    tensor_val = tensor_val.to(device)
    val_dataset = TensorDataset(tensor_val)

    # train model

    # load training and validation data
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size, shuffle=True)

    # load model weights
    vae.load_state_dict(torch.load(weights_path))

    # read old losses
    old_average_training_losses, old_average_validation_losses = read_losses()

    # do training and get new losses
    trainer = Trainer(vae, extra_epochs, train_loader, val_loader, device, loss_fn, optimizer, print_freq,
                      drop_infinity_from_loss_record_calc)
    new_average_training_losses, new_average_validation_losses = trainer.train_model()

    # concatenate old and new loss lists
    average_training_losses = old_average_training_losses + new_average_training_losses
    average_validation_losses = old_average_validation_losses + new_average_validation_losses

    # save weights
    torch.save(vae.state_dict(), weights_path)

    # record average training and validation losses per epoch
    with open('loss_record.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        headers = ["avg_train_loss", "avg_val_loss"]
        csv_writer.writerow(headers)

        for avg_train_loss, avg_val_loss in zip(average_training_losses, average_validation_losses):
            row = [avg_train_loss, avg_val_loss]
            csv_writer.writerow(row)

    # visualise average training and validation losses per epoch
    if visualise_training_and_validation_loss:
        losses_plot(average_training_losses, average_validation_losses, plot_loss_50_epoch_skip)

    # format all data
    data_tensor = torch.FloatTensor(data).view(-1, 1, data_sequence_size)
    data_tensor = data_tensor.to(device)
    dataset = TensorDataset(data_tensor)

    # load all data
    val_loader = DataLoader(dataset,
                            batch_size=batch_size, shuffle=True)

    # visualise reconstruction if visualisation is on
    if visualise_scatter:
        reconstruction_scatter_plot(vae, data, val_loader, show_y_equals_x, data_type, drop_scatter_outliers)

    plt.show()