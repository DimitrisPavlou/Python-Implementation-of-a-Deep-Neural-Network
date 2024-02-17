# Neural-Network-from-scratch-in-Python
I developed a neural network in Python for classificication and regression problems. The whole project follows an object oriented approach. There is definetely room from improvement and there will be additions to the project (an rbf layer class for example , or other optimizers). 

* Layer \n
  This files contains a basic implementation of a fully connected layer. The initialization of the weights is based on gaussian distribution 

* Optimizers \n
  This files contains the implementations of the SGD and Adam optimzer. Both optimizers have methods called pre_update and post_update which are called every epoch and a     main update_params method which is called every time we update parameters 

* ActivationFunctions  
  This files contains the typical activation functions : ReLu , Sigmoid , Tanh , Softmax

* Loss
  This files contains the implementations of the MSE loss and Categorical Cross Entropy Loss functions . It also contains a class for a common object for the combo Softmax   Activation function and Categorical Cross Entropy loss function. More details about this can be found here : 
  https://www.mldawn.com/back-propagation-with-cross-entropy-and-softmax/

* NeuralNetwork 
  This files contains the main function for the training of the network called train and 3 helper functions , create_batch , forward_pass , backword_pass. The batches are    created randomly. In the training function theres it the option to pass a validation set as well as the training set .There is also the option to stop the training         prematurely if the validation loss starts to increase.

* test
  In this file we perform a test run on the network on the fashion mnist dataset from keras. The accuracy we got was 88%

