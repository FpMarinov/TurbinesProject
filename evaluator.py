import numpy as np

class Evaluator(object):
    
    def __init__(self, eval_method):
        self.eval_method = eval_method
        self.losses = []

    def update(self, targets, predictions):
        loss = self.eval_method(predictions[0], predictions[1], predictions[2], targets)
        loss = loss.cpu().detach().item()
        self.losses.append(loss)

    def log(self):
        arr = np.array(self.losses)
        average_loss = np.mean(arr[np.isfinite(arr)])
        print("Average Validation Loss: {}".format(average_loss))
        return average_loss
