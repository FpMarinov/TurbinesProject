import torch
import matplotlib.pyplot as plt
from ReaderWriter import read_data_lists
from VAE import data_loader, VAE

latent_dimensions_thrust_torque = 1
weights_path_thrust = "./vae_net_thrust.pth"
weights_path_torque = "./vae_net_torque.pth"
seed = 1
lr = 1e-4
validation_data_fraction = 0.2
print_freq = 10

if __name__ == "__main__":
    # get data
    velocity_list, thrust_list, torque_list = read_data_lists()
    velocity_list = [x * 10 for x in velocity_list]
    thrust_list = [x / 10 for x in thrust_list]

    # set seed
    torch.manual_seed(seed)

    # setup neural network
    vae_thrust = VAE(latent_dimensions_thrust_torque)
    vae_torque = VAE(latent_dimensions_thrust_torque)

    # load model weights
    vae_thrust.load_state_dict(torch.load(weights_path_thrust))
    vae_torque.load_state_dict(torch.load(weights_path_torque))

    # choose device(cpu or gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vae_thrust.to(device)
    vae_torque.to(device)
