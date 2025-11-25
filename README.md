# 🧠 Deep Neural Network from Scratch (NumPy)

This project is a **from-scratch implementation of a Multilayer Perceptron (MLP)** in Python for both **classification** and **regression** tasks.  
It follows an **object-oriented design** and relies only on the **NumPy** library.  

The project is designed to be educational and extensible — for example, future additions may include an **RBF layer** or **additional optimizers**.  

---

## 📂 Project Structure

---

## 🔹 Layers (`nn/layers.py`)

Implements a fully connected **Dense layer** with:

- Forward pass  
- Backward pass  
- Parameter storage for optimizers  
- Weight initialization via `utils/weight_init.py`  
  (Gaussian, Xavier, He, etc.)

---

## 🔹 Activation Functions (`nn/activations.py`)

Implements the most common activation functions:

- **ReLU**  
- **Sigmoid**  
- **Tanh**  
- **Softmax**

Each activation class includes both `forward()` and `backward()`.

---

## 🔹 Loss Functions (`nn/loss_functions.py`)

Includes:

- **Mean Squared Error (MSE)**
- **Categorical Cross Entropy**
- **Softmax + Cross Entropy (Combined)**  
  → Numerically stable for classification

---

## 🔹 Optimizers (`optim/optimizers.py`)

Optimization algorithms implemented from scratch:

- **SGD**
- **Adam**
- **RMSProp**
- **LRScheduler(step, cosine, plateau)**

---

## 🔹 Training Engine (`training/trainer.py`)

Centralized training loop:

- Batch creation  
- Forward & backward passes  
- Parameter updates  
- Validation support  
- Callback system (via `callbacks.py`)  
  - Early stopping  
  - Logging  
  - Custom hooks  

---

## 🔹 Data Loader (`data/dataloader.py`)

Simple data loading utilities:

- Shuffling  
- Mini-batch generation  
- Dataset preparation helpers  
- PyTorch Style DataLoader 
---

## 🔹 Utilities (`utils/`)

Contains helper modules:

- `metrics.py` → Accuracy, etc.  
- `weight_init.py` → Weight initialization schemes (He, Xavier)
- Additional general utilities

---

## 🔹 Example Notebook (`train.ipynb`)

Demonstrates:

- Loading a dataset (e.g., Fashion-MNIST)  
- Defining the model  
- Training with validation  
- Evaluating performance  

Achieves **~90% accuracy** on Fashion-MNIST.

---

## ✨ Key Features

- Full **forward + backward** implementation  
- Completely modular architecture  
- Easy to extend with new layers, activations, losses, or optimizers  
- Works for **classification** and **regression**  
- Clean, readable structure ideal for educational purposes

---

## 📜 License

MIT License

---

