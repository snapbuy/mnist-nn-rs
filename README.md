# MNIST Neural Network in Rust
This is a from-scratch implementation of a feedforward neural network in Rust, built without using high-level machine learning libraries. 

## Results for 200 iterations and learning rate = 0.1
```console
DATA: 784, 60000
LABELS: 1, 60000
[[5, 0, 4, 1, 9, ..., 8, 3, 5, 6, 8]]

Iteration: 0
Accuracy: 10.86%

Iteration: 50
Accuracy: 56.97%

Iteration: 100
Accuracy: 69.91%

Iteration: 150
Accuracy: 75.45%

Iteration: 200
Accuracy: 78.56%
```

## Results for 500 iterations and learning rate = 0.1
```console
DATA: 784, 60000
LABELS: 1, 60000
[[5, 0, 4, 1, 9, ..., 8, 3, 5, 6, 8]]

Iteration: 0
Accuracy: 12.46%

Iteration: 50
Accuracy: 47.05%

Iteration: 100
Accuracy: 61.53%

Iteration: 150
Accuracy: 69.01%

Iteration: 200
Accuracy: 73.28%

Iteration: 250
Accuracy: 76.48%

Iteration: 300
Accuracy: 78.93%

Iteration: 350
Accuracy: 80.81%

Iteration: 400
Accuracy: 82.38%

Iteration: 450
Accuracy: 83.53%

Iteration: 500
Accuracy: 84.48%
```

## It demonstrates:

- Manual forward and backward propagation
- Use of ReLU and softmax activation functions
- One-hot encoding
- Gradient descent for training
- Accuracy evaluation
- Model parameter export to CSV using polars

## 🔧 Dependencies
- ndarray (store 2d array of data)
- ndarray-rand (generate intial random weights(w) and biases(b))
- polars (to read write data in csv)

## 🧠 Model Overview
- 1 Hidden layer neural network
- Input: 784-dimensional MNIST images
- Hidden layer: 10 neurons with ReLU
- Output layer: 10 neurons with softmax

## 📂 Structure
- `main.rs`: Training loop and evaluation
- `lib.rs`: Core model logic — forward, backward, update, softmax, etc.
- `final_config/`: Stores final weights and biases
- `mnistdata/`: Contains input dataset

## 📦 Dataset
Make sure the MNIST dataset is placed in mnistdata/.

