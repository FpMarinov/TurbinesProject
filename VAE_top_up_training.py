from sklearn.model_selection import train_test_split
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import matplotlib.pyplot as plt
from trainer import Trainer
from ReaderWriter import read_data_lists, read_losses, write_losses
from Plotter import losses_plot, reconstruction_scatter_plot
from VAE import loss_fn, data_loader, get_data, VAE

data_type = "torque"
extra_epochs = 13
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
    data = get_data()

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
                                            train_size=1 - validation_data_fraction, shuffle=False)

    # train model

    # load training and validation data
    train_loader = data_loader(data_train, device)
    val_loader = data_loader(data_val, device)

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
    write_losses(average_training_losses, average_validation_losses)

    # visualise average training and validation losses per epoch
    if visualise_training_and_validation_loss:
        losses_plot(average_training_losses, average_validation_losses, plot_loss_50_epoch_skip)

    # load all data
    val_loader = data_loader(data, device)

    # visualise reconstruction if visualisation is on
    if visualise_scatter:
        reconstruction_scatter_plot(vae, data, val_loader, show_y_equals_x, data_type, drop_scatter_outliers)

    plt.show()
