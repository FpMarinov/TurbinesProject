import torch
from torch import nn


class PredictionTrainer:
    """

    """

    def __init__(self, encoder1, encoder2, decoder, num_epochs, train_loader_enc1, train_loader_enc2, train_loader_pred,
                 val_loader_enc1, val_loader_enc2, val_loader_pred,
                 device, optimizer):
        """

        """
        self.loss_criterion = nn.MSELoss()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder = decoder
        self.num_epochs = num_epochs
        self.train_loader_enc1 = train_loader_enc1
        self.train_loader_enc2 = train_loader_enc2
        self.train_loader_pred = train_loader_pred
        self.val_loader_enc1 = val_loader_enc1
        self.val_loader_enc2 = val_loader_enc2
        self.val_loader_pred = val_loader_pred
        self.device = device
        self.optimizer = optimizer
        self.epoch = 0

    def train_model(self):
        """

        """
        self.encoder1.to(self.device)
        self.encoder2.to(self.device)
        self.decoder.to(self.device)

        average_training_losses = []
        average_validation_losses = []

        # epochs loop
        while self.epoch < self.num_epochs:
            self.decoder.train()

            training_losses_in_epoch = []

            # iterations loop
            for inputs1, inputs2, targets in zip(self.train_loader_enc1, self.train_loader_enc2,
                                                 self.train_loader_pred):
                # get encoded inputs and targets
                encoded_inputs_tensor = encode_inputs(inputs1, inputs2, self.encoder1, self.encoder2, self.device)
                targets = targets[0]

                # zero gradient and get outputs
                self.optimizer.zero_grad()
                outputs = self.decoder(encoded_inputs_tensor)

                # calculate loss and do backpropagation
                loss = self.loss_criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # add training loss to list
                loss_item = loss.cpu().detach().item()
                training_losses_in_epoch.append(loss_item)

            # print and add average training loss for epoch to list
            average_training_loss = sum(training_losses_in_epoch) / len(training_losses_in_epoch)
            average_training_losses.append(average_training_loss)
            print("Epoch {}: Average Training Loss: {}".format(self.epoch, average_training_loss))

            # print and add average validation loss for epoch to list
            average_validation_loss = self.eval_model()
            average_validation_losses.append(average_validation_loss)
            print("Epoch {}: Average Validation Loss: {}".format(self.epoch, average_validation_loss))

            # increment epoch
            self.epoch += 1

        return average_training_losses, average_validation_losses

    def eval_model(self):
        """

        """
        self.decoder.eval()

        validation_losses_in_epoch = []

        with torch.no_grad():
            # validation iterations loop
            for inputs1, inputs2, targets in zip(self.train_loader_enc1, self.train_loader_enc2,
                                                 self.train_loader_pred):
                # get encoded inputs and targets
                encoded_inputs_tensor = encode_inputs(inputs1, inputs2, self.encoder1, self.encoder2, self.device)
                targets = targets[0]

                # get outputs
                outputs = self.decoder(encoded_inputs_tensor)

                # calculate loss and add to list
                loss = self.loss_criterion(outputs, targets)
                loss_item = loss.cpu().detach().item()
                validation_losses_in_epoch.append(loss_item)

        # calculate average validation loss for epoch
        average_validation_loss = sum(validation_losses_in_epoch) / len(validation_losses_in_epoch)

        return average_validation_loss


def encode_inputs(inputs1, inputs2, encoder1, encoder2, device):
    """

    """
    # get inputs
    inputs1 = inputs1[0]
    inputs2 = inputs2[0]

    # encode inputs into means and logs of variances
    z_mean1, z_logvar1 = encoder1(inputs1)
    z_mean2, z_logvar2 = encoder2(inputs2)

    encoded_inputs_list = []

    # put the means and logs of variances together
    for x0, x1, x2, x3 in zip(z_mean1, z_logvar1,
                              z_mean2, z_logvar2):
        x0 = x0.item()
        x1 = x1.item()
        x2 = x2.item()
        x3 = x3.item()

        sequence = [x0, x1, x2, x3]

        encoded_inputs_list.append(sequence)

    # transform encoded inputs into tensor on device
    encoded_inputs_tensor = torch.tensor(encoded_inputs_list)
    encoded_inputs_tensor = encoded_inputs_tensor.to(device)

    return encoded_inputs_tensor
