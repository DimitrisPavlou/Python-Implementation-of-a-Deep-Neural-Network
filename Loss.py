import numpy as np
from ActivationFunctions import Softmax


#mean square error loss class
class MSE() :

    def forward(self, y_pred , y_true) :
        sample_losses = 0.5*np.sum((y_true - y_pred)**2 , axis = -1)
        return sample_losses

    def backward(self , dvalues , y_true , combinded = False) :
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2*(y_true - dvalues)/outputs
        self.dinputs = self.dinputs/samples
    
    def calculate(self, output, y):
        sample_losses = self.forward(output , y) 
        data_loss = np.mean(sample_losses)
        return data_loss

#===============================================================================

#categorical cross entropy loss class

class CategoricalCrossEntropy() :

    def forward(self , y_pred , y_true) :
        samples = len(y_pred)
        #clipping the y_pred so as not to have log(0) and log(1)
        y_pred_clipped = np.clip(y_pred , 1e-7 , 1-1e-7)
        if len(y_true.shape) == 1 :
            correct_confidences = y_pred_clipped[range(samples) , y_true]
        elif len(y_true.shape) == 2 :
            correct_confidences = np.sum(y_pred_clipped*y_true , axis = 1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
    
    def calculate(self, output, y):
        sample_losses = self.forward(output , y) 
        data_loss = np.mean(sample_losses)
        return data_loss

#===============================================================================

#create a class that acts as an output layer activation function and loss function
#at the same time to combine the softmax activation function and categorical cross
#entropy loss function for faster calculations


class Softmax_CategoricalCrossEntropy_Combined():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)


    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
