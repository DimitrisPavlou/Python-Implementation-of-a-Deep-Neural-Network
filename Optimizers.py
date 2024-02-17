import numpy as np


class SGD :

    def __init__(self , learning_rate = 0.01 , lr_decay = 0 , momentum = 0 , momentum_decay = 0) :
        #initialize the parameters of the optimizer
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.lr_decay = lr_decay
        self.iterations = 0
        self.momentum = momentum
        self.momentum_decay = momentum_decay

    def pre_update(self) :
        #apply the learning rate decay if specified
        if self.lr_decay :
            if self.current_lr > 1e-4 :
                self.current_lr *= 1/(1 + self.lr_decay * self.iterations)
            else :
                self.current_lr = 1e-4
        #apply the momentum decay if specified
        if self.momentum_decay :
            if self.momentum > 0.05 :
                self.momentum *= (1 - self.momentum_decay * self.iterations)
            else :
                self.momentum = 0

    def update_params(self , layer) :
        #update parameters in the case of momentum
        if self.momentum :
            weight_updates = self.momentum * layer.weights_momentum - self.current_lr * layer.dweights
            layer.weights_momentum = weight_updates

            bias_updates = self.momentum * layer.biases_momentum - self.current_lr * layer.dbiases
            layer.biases_momentum = bias_updates
        #update parameters without the use of momemntum
        else :
            weight_updates = - self.current_lr * layer.dweights
            bias_updates = - self.current_lr * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update(self) :
        self.iterations += 1

#===============================================================================

class Adam :

    def __init__(self , learning_rate = 0.001 , beta_1 = 0.9 , beta_2 = 0.999 , epsilon = 1e-7 , decay = 0) :
        #initialize parameters of the optimizer
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2


    def pre_update(self) :

        if self.decay :
            if self.current_lr > 1e-4 :
                self.current_lr *= 1/(1 + self.decay * self.iterations)
            else :
                self.current_lr = 1e-4

    def update_params(self , layer) :
        #params from sgd

        #momentum weights and biases
        layer.weights_momentum = self.beta_1 * layer.weights_momentum + \
                                 (1 - self.beta_1) * layer.dweights
        layer.biases_momentum =  self.beta_1 * layer.biases_momentum + \
                                 (1 - self.beta_1) * layer.dbiases

        #corrected momentum weights and biases

        weights_momentum_corrected = layer.weights_momentum/(1 - (self.beta_1)**(self.iterations + 1))
        biases_momentum_corrected = layer.biases_momentum/(1 - (self.beta_1)**(self.iterations + 1))

        #params from rmsprop

        layer.weights_cache = self.beta_2 * layer.weights_cache + (1 - self.beta_2) * layer.dweights**2
        layer.biases_cache = self.beta_2 * layer.biases_cache + (1 - self.beta_2) * layer.dbiases**2

        #corrected cache weights and biases
        weights_cache_corrected = layer.weights_cache/(1 - (self.beta_2)**(self.iterations + 1))
        biases_cache_corrected = layer.biases_cache /(1 - (self.beta_2)**(self.iterations + 1))

        #update params

        layer.weights += -self.current_lr * weights_momentum_corrected / (np.sqrt(weights_cache_corrected + self.epsilon))
        layer.biases += -self.current_lr * biases_momentum_corrected / (np.sqrt(biases_cache_corrected + self.epsilon))

    def post_update(self) :
        self.iterations += 1
