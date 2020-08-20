"""
Contains the prediction decoder class for velocity, PredictionDecoderVelocity.
It is used when PredictionDecoder.py is run with data_to_predict_type = "velocity".

Based on:
    File Name: main.py
    Developed by: Nikolas Pitsillos, PhD Candidate in Computer Vision and Autonomous Systems @ UofG
    Taken from: https://github.com/npitsillos/mnist-vae/blob/master/main.py
    Described at: https://npitsillos.github.io/posts/2020/05/mnistvae/
"""

from torch import nn
import torch.nn.functional as F
from VAE import latent_dimensions, data_sequence_size

# variables that should not be changed
fully_connected_unit_size = 400
convolution_channel_size_1 = 4
convolution_channel_size_2 = 4
convolution_channel_size_3 = 4
convolution_kernel = 3


class PredictionDecoderVelocity(nn.Module):
    """
    Decoder used for predicting velocity from the encoded means and log variances
    of thrust and torque.
    """

    def __init__(self):
        """
        Initializes internal PredictionDecoderVelocity state.
        """
        super(PredictionDecoderVelocity, self).__init__()

        # latent space transformation
        self.fc_lat = nn.Linear(4 * latent_dimensions, fully_connected_unit_size)

        # fully connected transformation
        self.fc1 = nn.Linear(fully_connected_unit_size, convolution_channel_size_1 * data_sequence_size)

        # convolution
        self.conv1 = nn.Conv1d(convolution_channel_size_1, convolution_channel_size_2, convolution_kernel, 1, 1)
        self.conv2 = nn.Conv1d(convolution_channel_size_2, convolution_channel_size_3, convolution_kernel, 1, 1)
        self.conv3 = nn.Conv1d(convolution_channel_size_3, 1, convolution_kernel, 1, 1)

    def forward(self, z_input):
        """
        Defines the computation performed at every call.

        Args:
            z_input (Tensor): input tensor.

        Returns:
            Tensor: decoded input.
        """
        # latent space transformation
        x = self.fc_lat(z_input)
        x = F.relu(x)

        # fully connected transformation
        x = self.fc1(x)
        x = F.relu(x)

        # reformat
        x = x.view(z_input.size()[0], convolution_channel_size_1, data_sequence_size)

        # convolution
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        return x



