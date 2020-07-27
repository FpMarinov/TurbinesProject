import torch
import matplotlib.pyplot as plt
from ReaderWriter import read_losses, write_losses
from Plotter import losses_plot, reconstruction_scatter_plot
from VAE import data_loader, get_data, setup


data_type = "thrust"
extra_epochs = 3
visualise_scatter = True
drop_scatter_outliers = False
show_y_equals_x = True
visualise_training_and_validation_loss = True
plot_loss_50_epoch_skip = True
weights_path = "./vae_net_%s.pth" % data_type


if __name__ == "__main__":
    # get data
    data = get_data(data_type)

    # setup
    device, vae, trainer = setup(extra_epochs)

    # train model

    # load model weights
    vae.load_state_dict(torch.load(weights_path))

    # read old losses
    old_average_training_losses, old_average_validation_losses = read_losses()

    # do training and get new losses
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
