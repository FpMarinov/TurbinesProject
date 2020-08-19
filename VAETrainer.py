import torch


class VAETrainer:
    """

    """

    def __init__(self, model, num_epochs, train_loader, val_loader,
                 device, loss_criterion, optimizer):
        """

        """
        self.loss_criterion = loss_criterion
        self.model = model
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer
        self.epoch = 0

    def train_model(self):
        """

        """
        self.model.to(self.device)

        average_total_training_losses = []
        average_total_validation_losses = []
        average_mse_training_losses = []
        average_mse_validation_losses = []

        # epochs loop
        while self.epoch < self.num_epochs:
            self.model.train()

            total_training_losses_in_epoch = []
            mse_training_losses_in_epoch = []

            # iterations loop
            for inputs_targets in self.train_loader:
                # get inputs and targets
                inputs_targets = inputs_targets[0]

                # zero gradient and get outputs
                self.optimizer.zero_grad()
                outputs = self.model(inputs_targets)

                # calculate loss and do backpropagation
                total_loss, mse_loss = self.loss_criterion(outputs[0], outputs[1], outputs[2], inputs_targets)
                total_loss.backward()
                self.optimizer.step()

                # add training loss to list
                total_loss_item = total_loss.cpu().detach().item()
                total_training_losses_in_epoch.append(total_loss_item)
                mse_loss_item = mse_loss.cpu().detach().item()
                mse_training_losses_in_epoch.append(mse_loss_item)

            # calculate, print and add average training loss for epoch to list
            average_total_training_loss = sum(total_training_losses_in_epoch) / len(total_training_losses_in_epoch)
            average_total_training_losses.append(average_total_training_loss)
            print("Epoch {}: Average Total Training Loss: {}".format(self.epoch, average_total_training_loss))
            average_mse_training_loss = sum(mse_training_losses_in_epoch) / len(mse_training_losses_in_epoch)
            average_mse_training_losses.append(average_mse_training_loss)
            print("Epoch {}: Average MSE Training Loss: {}".format(self.epoch, average_mse_training_loss))

            # calculate, print and add average validation loss for epoch to list
            average_total_validation_loss, average_mse_validation_loss = self.eval_model()
            average_total_validation_losses.append(average_total_validation_loss)
            print("Epoch {}: Average Total Validation Loss: {}".format(self.epoch, average_total_validation_loss))
            average_mse_validation_losses.append(average_mse_validation_loss)
            print("Epoch {}: Average MSE Validation Loss: {}".format(self.epoch, average_mse_validation_loss))

            # increment epoch
            self.epoch += 1

        return average_total_training_losses, average_total_validation_losses, \
            average_mse_training_losses, average_mse_validation_losses

    def eval_model(self):
        """

        """
        self.model.eval()

        total_validation_losses_in_epoch = []
        mse_validation_losses_in_epoch = []

        with torch.no_grad():
            # validation iterations loop
            for inputs_targets in self.val_loader:
                # get inputs and targets
                inputs_targets = inputs_targets[0]

                # get outputs
                outputs = self.model(inputs_targets)

                # calculate loss and add to list
                total_loss, mse_loss = self.loss_criterion(outputs[0], outputs[1], outputs[2], inputs_targets)
                total_loss_item = total_loss.cpu().detach().item()
                total_validation_losses_in_epoch.append(total_loss_item)
                mse_loss_item = mse_loss.cpu().detach().item()
                mse_validation_losses_in_epoch.append(mse_loss_item)

        # calculate average validation loss for epoch
        average_total_validation_loss = sum(total_validation_losses_in_epoch) / len(total_validation_losses_in_epoch)
        average_mse_validation_loss = sum(mse_validation_losses_in_epoch) / len(mse_validation_losses_in_epoch)

        return average_total_validation_loss, average_mse_validation_loss
