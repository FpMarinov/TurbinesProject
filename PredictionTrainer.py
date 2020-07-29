import time
import torch
from utils import Logger
from evaluator import Evaluator
import numpy as np


class Trainer:

    def __init__(self, encoder1, encoder2, decoder, num_epochs, train_loader_enc1, train_loader_enc2, train_loader_pred,
                 val_loader_enc1, val_loader_enc2, val_loader_pred,
                 device, loss_criterion, optimizer, print_freq, drop_infinity_from_loss_record_calc):
        self.drop_infinity_from_loss_record_calc = drop_infinity_from_loss_record_calc
        self.loss_criterion = loss_criterion
        self.evaluator = Evaluator(self.loss_criterion)
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
        self.print_freq = print_freq
        self.optimizer = optimizer
        self.epoch = 0


    def train_model(self):
        self.encoder1.to(self.device)
        self.encoder2.to(self.device)
        self.decoder.to(self.device)

        average_training_losses = []
        average_validation_losses = []

        while self.epoch < self.num_epochs:
            self.decoder.train()

            training_losses_in_epoch = []

            for inputs1, inputs2, targets in zip(self.train_loader_enc1, self.train_loader_enc2, self.train_loader_pred):
                inputs1 = inputs1[0]
                inputs2 = inputs2[0]
                targets = targets[0]

                z_mean1, z_logvar1 = self.encoder1(inputs1)
                z_mean2, z_logvar2 = self.encoder2(inputs2)

                encoded_inputs_list = []

                for x0, x1, x2, x3 in zip(z_mean1, z_logvar1,
                                          z_mean2, z_logvar2):
                    x0 = x0.item()
                    x1 = x1.item()
                    x2 = x2.item()
                    x3 = x3.item()

                    sequence = [x0, x1, x2, x3]

                    encoded_inputs_list.append(sequence)

                encoded_inputs_tensor = torch.tensor(encoded_inputs_list)
                encoded_inputs_tensor = encoded_inputs_tensor.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.decoder(encoded_inputs_tensor)

                ########### change #########################
                loss = self.loss_criterion(outputs[0], outputs[1], outputs[2], inputs_targets)
                loss.backward()
                self.optimizer.step()

                loss_item = loss.cpu().detach().item()

                self.logger.update(loss=loss_item)
                self.logger.update(lr=self.optimizer.param_groups[0]["lr"])

                training_losses_in_epoch.append(loss_item)
                ########### change #########################

            arr = np.array(training_losses_in_epoch)
            if self.drop_infinity_from_loss_record_calc:
                average_training_loss = np.mean(arr[np.isfinite(arr)])
            else:
                average_training_loss = np.mean(arr)
            average_training_losses.append(average_training_loss)

            average_validation_loss = self.eval_model()
            average_validation_losses.append(average_validation_loss)

            self.epoch += 1

        return average_training_losses, average_validation_losses

    def eval_model(self):
        self.encoder1.eval()
        # Evaluate model
        with torch.no_grad():
            for inputs_targets in self.logger.log(self.val_loader_enc1, self.print_freq, "Validation:", training=False):
                inputs_targets = inputs_targets[0]

                model_time = time.time()
                outputs = self.encoder1(inputs_targets)
                model_time = time.time() - model_time

                evaluator_time = time.time()
                self.evaluator.update(inputs_targets, outputs)
                evaluator_time = time.time() - evaluator_time

                self.logger.update(model_time=model_time, evaluator_time=evaluator_time)

        average_validation_loss = self.evaluator.log(self.drop_infinity_from_loss_record_calc)
        return average_validation_loss
