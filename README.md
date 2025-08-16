# 🧠 Deep Neural Network from Scratch (NumPy)

This project is a **from-scratch implementation of a Deep Neural Network (DNN)** in Python for both **classification** and **regression** tasks.  
It follows an **object-oriented design** and relies only on the **NumPy** library.  

The project is designed to be educational and extensible — for example, future additions may include an **RBF layer** or **additional optimizers**.  

---

## 📂 Project Structure

### 🔹 Layer
- Implements a fully connected (dense) layer.  
- Weights are initialized using a **Gaussian distribution**.  

### 🔹 Optimizers
- Includes implementations of **SGD** and **Adam**.  
- Each optimizer defines:
  - `pre_update()` → called at the start of each epoch  
  - `update_params()` → updates parameters on every batch  
  - `post_update()` → called at the end of each epoch  

### 🔹 ActivationFunctions
- Implements standard activation functions:  
  - **ReLU**  
  - **Sigmoid**  
  - **Tanh**  
  - **Softmax**  

### 🔹 Loss
- Implements loss functions:  
  - **Mean Squared Error (MSE)**  
  - **Categorical Cross Entropy**  
- Includes a combined **Softmax + Cross Entropy** class for stable backpropagation.  
- More details: [Backpropagation with Cross Entropy and Softmax](https://www.mldawn.com/back-propagation-with-cross-entropy-and-softmax/)  

### 🔹 NeuralNetwork
- Core training logic with methods:  
  - `train()` → main training loop  
  - `create_batch()` → random mini-batch generation  
  - `forward_pass()` → computes outputs  
  - `backward_pass()` → propagates gradients  
- Supports:  
  - Training with/without a validation set  
  - Early stopping if validation loss increases  

### 🔹 test
- Test run using the **Fashion-MNIST** dataset from Keras.  
- Achieved **~88% accuracy**.  

---

## ✨ Key Features
- Fully implemented **forward and backward passes**  
- Customizable architecture with multiple layers and activations  
- Works for both **classification** and **regression** problems  
- Educational code structure, easy to extend with new layers/optimizers  
