import sys
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from Plotter import losses_plot, prediction_reconstruction_scatter_plot
from PredictionDecoderVelocity import PredictionDecoderVelocity
from PredictionTrainer import PredictionTrainer
from ReaderWriter import read_data_lists, write_general_losses
from VAE import data_loader, VAE, latent_dimensions, data_sequence_size, seed, validation_data_fraction, lr, Decoder, \
    convolution_channel_size_4, fully_connected_unit_size

data_to_predict_type = "thrust"
mode = "train"
epochs = 3
plot_loss_1_epoch_skip = True
plot_loss_50_epoch_skip = False
sampling = False

weights_path_thrust = "./vae_net_thrust.pth"
weights_path_torque = "./vae_net_torque.pth"
weights_path_velocity = "./vae_net_velocity.pth"
weights_path_decoder = "./vae_net_prediction_decoder_%s.pth" % data_to_predict_type


class PredictionDecoder(Decoder):

    def __init__(self):
        super(PredictionDecoder, self).__init__(latent_dimensions, convolution_channel_size_4 * data_sequence_size)

        # latent space transformation
        if sampling:
            self.fc_lat = nn.Linear(2 * latent_dimensions, fully_connected_unit_size)
        else:
            self.fc_lat = nn.Linear(4 * latent_dimensions, fully_connected_unit_size)


def get_data_and_weights():
    # get all data
    velocity_list, thrust_list, torque_list = read_data_lists()

    # multiply velocities by 10
    velocity_list = [x * 10 for x in velocity_list]

    # divide thrusts by 10
    thrust_list = [x / 10 for x in thrust_list]

    # choose correct data and weights
    if data_to_predict_type == "velocity":
        data_to_encode1 = thrust_list
        data_to_encode2 = torque_list
        data_to_predict = velocity_list

        weights_path_vae1 = weights_path_thrust
        weights_path_vae2 = weights_path_torque
    elif data_to_predict_type == "torque":
        data_to_encode1 = thrust_list
        data_to_encode2 = velocity_list
        data_to_predict = torque_list

        weights_path_vae1 = weights_path_thrust
        weights_path_vae2 = weights_path_velocity
    elif data_to_predict_type == "thrust":
        data_to_encode1 = torque_list
        data_to_encode2 = velocity_list
        data_to_predict = thrust_list

        weights_path_vae1 = weights_path_torque
        weights_path_vae2 = weights_path_velocity
    else:
        sys.exit("Incorrect data type.")

    return data_to_encode1, data_to_encode2, data_to_predict, weights_path_vae1, weights_path_vae2


def setup(data_to_encode1, data_to_encode2, data_to_predict, weights_path_vae1, weights_path_vae2):
    # set seed
    torch.manual_seed(seed)

    # setup neural networks and optimizer
    vae1 = VAE(latent_dimensions)
    vae2 = VAE(latent_dimensions)
    if data_to_predict_type == "velocity":
        decoder = PredictionDecoderVelocity(sampling)
    else:
        decoder = PredictionDecoder()
    optimizer = Adam(decoder.parameters(), lr=lr)

    # load model weights
    vae1.load_state_dict(torch.load(weights_path_vae1))
    vae2.load_state_dict(torch.load(weights_path_vae2))

    # get encoders
    encoder1 = vae1.encoder
    encoder2 = vae2.encoder

    # set encoders to evaluation
    encoder1.eval()
    encoder2.eval()

    # choose device(cpu or gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoder1.to(device)
    encoder2.to(device)
    decoder.to(device)

    if mode == "train":
        # split data into training and validation data
        training_data_to_encode1, validation_data_to_encode1 = train_test_split(data_to_encode1,
                                                                                test_size=validation_data_fraction,
                                                                                train_size=1 - validation_data_fraction,
                                                                                shuffle=False)
        training_data_to_encode2, validation_data_to_encode2 = train_test_split(data_to_encode2,
                                                                                test_size=validation_data_fraction,
                                                                                train_size=1 - validation_data_fraction,
                                                                                shuffle=False)
        training_data_to_predict, validation_data_to_predict = train_test_split(data_to_predict,
                                                                                test_size=validation_data_fraction,
                                                                                train_size=1 - validation_data_fraction,
                                                                                shuffle=False)

        # load training and validation data
        training_loader_to_encode1 = data_loader(training_data_to_encode1, device, shuffle=False)
        validation_loader_to_encode1 = data_loader(validation_data_to_encode1, device, shuffle=False)

        training_loader_to_encode2 = data_loader(training_data_to_encode2, device, shuffle=False)
        validation_loader_to_encode2 = data_loader(validation_data_to_encode2, device, shuffle=False)

        training_loader_to_predict = data_loader(training_data_to_predict, device, shuffle=False)
        validation_loader_to_predict = data_loader(validation_data_to_predict, device, shuffle=False)

        trainer = PredictionTrainer(encoder1, encoder2, decoder, epochs,
                                    training_loader_to_encode1, training_loader_to_encode2, training_loader_to_predict,
                                    validation_loader_to_encode1, validation_loader_to_encode2,
                                    validation_loader_to_predict, device, optimizer, sampling)
    else:
        trainer = None

    return device, encoder1, encoder2, decoder, trainer


if __name__ == "__main__":
    # get data and weights
    data_to_encode1, data_to_encode2, data_to_predict, weights_path_vae1, weights_path_vae2 = get_data_and_weights()

    # setup
    device, encoder1, encoder2, decoder, trainer = setup(data_to_encode1, data_to_encode2, data_to_predict,
                                                         weights_path_vae1, weights_path_vae2)

    # train model if training is on
    if mode == "train":
        # do training and get losses
        average_training_losses, average_validation_losses = trainer.train_model()

        # save weights
        torch.save(decoder.state_dict(), weights_path_decoder)

        # record average training and validation losses per epoch
        write_general_losses(average_training_losses, average_validation_losses)

        # visualise average training and validation losses per epoch
        losses_plot(average_training_losses, average_validation_losses,
                    plot_1_epoch_skip=plot_loss_1_epoch_skip, plot_50_epoch_skip=plot_loss_50_epoch_skip)

    # load all data
    validation_loader_to_encode1 = data_loader(data_to_encode1, device, shuffle=False)
    validation_loader_to_encode2 = data_loader(data_to_encode2, device, shuffle=False)
    validation_loader_to_predict = data_loader(data_to_predict, device, shuffle=False)

    # load model weights and activate evaluation if training is off
    if mode != "train":
        decoder.load_state_dict(torch.load(weights_path_decoder))
        decoder.eval()

    # visualise reconstruction
    prediction_reconstruction_scatter_plot(encoder1, encoder2, decoder, device, data_to_predict,
                                           data_to_predict_type, validation_loader_to_encode1,
                                           validation_loader_to_encode2, sampling)

    plt.show()
