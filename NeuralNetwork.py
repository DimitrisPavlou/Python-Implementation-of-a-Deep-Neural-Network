import numpy as np
from Loss import CategoricalCrossEntropy , Softmax_CategoricalCrossEntropy_Combined
from ActivationFunctions import Softmax

#simple functions that takes X_train and outputs a batch of it
def create_batch(X ,y , batch_size) :
    # the size of the batch is defined by the batch_size variable and is created by 
    # randomly taking samples from the train set
    indices = np.random.randint(low = 0 , high = len(y) , size = (batch_size,))
    return X[indices] , y[indices]


#===============================================================================

#perform a forward pass of the network and stop either at the last layer which is
#an activation output layer or the second to last which is a dense
#output layer. The second case happens only when the activation layer has the softmax
#function as activation function and the loss function is categorical cross entropy
#since we created a combined activation-loss class for the combination of the two

def forward_pass(network , X_batch , index) :
    network[0].forward(X_batch)
    for i in range(1,index) :
        network[i].forward(network[i-1].output)

#===============================================================================

#perform backward pass , update parameters and use momentum if needed.Again the
#index variable here is used to signal that we have used the softmax-cross entropy
#combined class like before

def backward_pass(network , optimizer , index):
    for i in range(index-2 , -1 , -1) :
        network[i].backward(network[i+1].dinputs)

    for i in range(0,len(network) , 2) :
        optimizer.update_params(network[i])
#===============================================================================

#the main function for training the network

def train(network , X_train , y_train , loss_function , epochs, optimizer , batch_size , X_val = 0 , y_val = 0 ,early_stopping = False):

    #save the average train and validation errors 
    avg_train_errors = []
    validation_errors = []
    
    avg_train_error_epoch = 0
    flag = False
    index = len(network)
    batch_epoch = int(X_train.shape[0]/batch_size)

    #check if a validation set is provided 
    val_set = False 
    if len(X_val) != 0 and len(y_val) != 0: 
        val_set = True 


    #check if Categorical Cross Entropy loss function is combined with Softmax activation function (condition 1)
    if isinstance(loss_function , CategoricalCrossEntropy) and isinstance(network[-1] , Softmax):
        #if so use a common object for faster calculations
        loss_function = Softmax_CategoricalCrossEntropy_Combined()
        #the last object of the network list which is the Softmax activation layer is not used 
        #in this case so we go as far as the previous to last object which is a dense layer 
        index += -1
        flag = True

    for epoch in range(epochs) :
        batch_counter = 0
        #nessessary based on the implementation of the optimizer objects
        optimizer.pre_update()

        for batch_epoch in range(batch_epoch) :
            #create a batch with size batch_size
            (X_batch , y_batch) = create_batch(X_train , y_train , batch_size)

            #perform the forward pass of the network
            forward_pass(network , X_batch , index)
            #check if condition 1 holds 
            if flag:
                #calculate the loss and the gradient using the common object we created  
                loss = loss_function.forward(network[index-1].output , y_batch)
                loss_function.backward(loss_function.output , y_batch)
                network[index-1].backward(loss_function.dinputs)

            else :
                #calculate the loss and the gradient the typical way
                loss_function.forward(network[-1].output , y_batch)
                loss_function.backward(network[-1].output, y_batch)
                network[-1].backward(loss_function.dinputs)

            #perform the backward pass on the network
            backward_pass(network, optimizer , index)

            #again check if condition 1 holds and append the train loss to the appropriate list
            if flag :
                data_loss_test = loss_function.forward(network[index-1].output , y_batch)
                avg_train_error_epoch += data_loss_test
            else :
                data_loss_test = loss_function.calculate(network[-1].output, y_batch)
                avg_train_error_epoch += data_loss_test

            batch_counter += batch_size

        #nessessary based on the implementation of the optimizer objects
        optimizer.post_update()
        #calculate the average per epoch train loss
        avg_train_error_epoch /= batch_epoch
        avg_train_errors.append(avg_train_error_epoch)

        #test validation set if provided 
        if val_set : 
            #forward pass on validation set
            forward_pass(network, X_val ,index)
            #check if condition 1 holds
            if flag :
                #calculate the loss 
                data_loss = loss_function.forward(network[index-1].output , y_val)
                #calculate the predicted values
                predictions = np.argmax(loss_function.output , axis =1)
                validation_errors.append(data_loss)
            else :
                #calculate the loss 
                data_loss = loss_function.calculate(network[-1].output , y_val)
                #calculate the predicted values
                predictions = np.argmax(network[-1].output , axis = 1)
                validation_errors.append(data_loss)
            #if y is one-hot encoded then convert it to a vector
            if len(y_val.shape) == 2 :
                y = np.argmax(y_val , axis = 1)
            #calculate accuracy
            accuracy = np.mean(predictions == y)

            #early stopping condition
            if early_stopping and epoch > 1 :
                if validation_errors[epoch-1] < validation_errors[epoch] :
                    break
            
            #print the parameters
            print(f"epoch : {epoch+1} | " + f"Validation acc : {accuracy:.3f} | " + f"Validation loss : {data_loss:.5f} | " + f"Train loss : {data_loss_test:.5f} | " + \
                f"learning rate : {optimizer.current_lr:.5f} " ) #+ f"momentum : {optimizer.momentum}")
        
        if not val_set : 
            print(f"epoch : {epoch + 1} | " + f"Train Loss : {data_loss_test:.5f} | " + f"learning rate : {optimizer.current_lr:.5f}")

    return (avg_train_errors , validation_errors)

#===============================================================================

#function that takes as input the test set and the loss function and returns the
#accuracy and the loss of the trained network on the test set


def predict(network , X_test , y_test , loss_function) :
    #forward pass
    network[0].forward(X_test)
    for i in range(1 , len(network)) :
      network[i].forward(network[i-1].output)

    #making the predictions and calculating the loss and the test accuracy score
    predictions = np.argmax(network[-1].output , axis = 1)
    if len(y_test.shape) == 2 :
      y = np.argmax(y_test , axis = 1)
    accuracy = np.mean(predictions == y)
    loss = loss_function.calculate(network[-1].output , y_test)
    
    return (loss , accuracy)