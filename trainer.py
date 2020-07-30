import torch


class Trainer:

    def __init__(self, model, num_epochs, train_loader, val_loader,
                 device, loss_criterion, optimizer):
        self.loss_criterion = loss_criterion
        self.model = model
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer
        self.epoch = 0

    def train_model(self):
        self.model.to(self.device)

        average_training_losses = []
        average_validation_losses = []

        while self.epoch < self.num_epochs:
            self.model.train()

            training_losses_in_epoch = []

            for inputs_targets in self.train_loader:
                inputs_targets = inputs_targets[0]

                self.optimizer.zero_grad()

                outputs = self.model(inputs_targets)

                loss = self.loss_criterion(outputs[0], outputs[1], outputs[2], inputs_targets)
                loss.backward()
                self.optimizer.step()

                loss_item = loss.cpu().detach().item()
                training_losses_in_epoch.append(loss_item)

            average_training_loss = sum(training_losses_in_epoch) / len(training_losses_in_epoch)
            average_training_losses.append(average_training_loss)
            print("Epoch {}: Average Training Loss: {}".format(self.epoch, average_training_loss))

            average_validation_loss = self.eval_model()
            average_validation_losses.append(average_validation_loss)
            print("Epoch {}: Average Validation Loss: {}".format(self.epoch, average_validation_loss))

            self.epoch += 1

        return average_training_losses, average_validation_losses

    def eval_model(self):
        self.model.eval()

        validation_losses_in_epoch = []

        with torch.no_grad():
            for inputs_targets in self.val_loader:
                inputs_targets = inputs_targets[0]

                outputs = self.model(inputs_targets)

                loss = self.loss_criterion(outputs[0], outputs[1], outputs[2], inputs_targets)
                loss_item = loss.cpu().detach().item()
                validation_losses_in_epoch.append(loss_item)

        average_validation_loss = sum(validation_losses_in_epoch) / len(validation_losses_in_epoch)

        return average_validation_loss
