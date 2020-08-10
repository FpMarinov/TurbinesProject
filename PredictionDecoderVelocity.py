from torch import nn
import torch.nn.functional as F
from VAE import latent_dimensions, data_sequence_size


sampling = False

fully_connected_unit_size = 400
convolution_channel_size_1 = 4
convolution_channel_size_2 = 4
convolution_channel_size_3 = 4
convolution_kernel = 3


class PredictionDecoderVelocity(nn.Module):

    def __init__(self):
        super(PredictionDecoderVelocity, self).__init__()

        # latent space transformation
        if sampling:
            self.fc_lat = nn.Linear(2 * latent_dimensions, fully_connected_unit_size)
        else:
            self.fc_lat = nn.Linear(4 * latent_dimensions, fully_connected_unit_size)

        # fully connected transformation
        self.fc1 = nn.Linear(fully_connected_unit_size, convolution_channel_size_1 * data_sequence_size)

        # convolution
        self.conv1 = nn.Conv1d(convolution_channel_size_1, convolution_channel_size_2, convolution_kernel, 1, 1)
        self.conv2 = nn.Conv1d(convolution_channel_size_2, convolution_channel_size_3, convolution_kernel, 1, 1)
        self.conv3 = nn.Conv1d(convolution_channel_size_3, 1, convolution_kernel, 1, 1)

    def forward(self, z_input):
        # print("-1:", z_input.size())

        x = self.fc_lat(z_input)
        x = F.relu(x)
        # print("-2:", x.size())

        x = self.fc1(x)
        x = F.relu(x)
        # print("-3:", x.size())

        x = x.view(z_input.size()[0], convolution_channel_size_1, data_sequence_size)
        # print("-4:", x.size())

        x = self.conv1(x)
        x = F.relu(x)
        # print("-5:", x.size())

        x = self.conv2(x)
        x = F.relu(x)
        # print("-6:", x.size())

        x = self.conv3(x)
        x = F.relu(x)
        # print("-7:", x.size())

        return x



