import numpy as np

class ReLu :
    def forward(self , inputs) :
        #clip inputs to avoid overflow errors
        inputs = np.clip(inputs , -500 , 500)
        #take the output of a dense layer and apply the relu non linear function
        self.inputs = inputs
        self.output = np.maximum(0 , inputs)

    def backward(self, dvalues):
        # Since we need to modify original variable,
        # we make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

#===============================================================================


class Sigmoid:

    def forward(self , inputs) :
        #clip inputs to avoid overflow errors
        inputs = np.clip(inputs , -10 , 10)
        #take the output of a dense layer and apply the non linear function
        self.inputs = inputs
        self.output = 1/(1 + np.exp(-inputs))

    def backward(self , dvalues) :
        # Since we need to modify original variable,
        # we make a copy of values first
        self.dinputs = dvalues.copy()
        # calculate the gradient of sigmoid
        self.dinputs = dvalues*(1-self.output)*self.output

#===============================================================================

class Tanh:
    def forward(self , inputs) :
        #clip inputs to avoid overflow errors
        inputs = np.clip(inputs , -10 , 10)
        #take the output of a dense layer and apply the non linear function
        self.inputs = inputs
        self.output = np.tanh(inputs)

    def backward(self , dvalues) :
        # Since we need to modify original variable,
        # we make a copy of values first
        self.dinputs = dvalues.copy()
        # calculate the gradient of sigmoid
        self.dinputs = dvalues*(1-(self.output)**2)

#===============================================================================

class Softmax:

    def forward(self , inputs) :
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs , axis = 1 , keepdims=True))
        probabilities = exp_values/np.sum(exp_values , axis = 1 , keepdims=True)
        self.output = probabilities


    def backward(self, dvalues):
      # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
          # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
