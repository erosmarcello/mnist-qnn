# MNIST Quantum Neural Network (TorchQuantum)

A hybrid quantum-classical MNIST classifier built with PyTorch + TorchQuantum.

## The Pertinents

- Loads and preprocesses the MNIST handwritten digits dataset
- Defines a hybrid model with a classical feature projector + variational quantum circuit
- Saves and loads model weights
- Evaluates performance on test data

## Stack
- Python 3.10
- PyTorch
- TorchQuantum
- Jupyter Notebook (for experimenting and quantum vs. classical models)
- VS Code / Cursor 
- Git + GitHub

## Files
- `mnist_nn.py`: Main training and inference code (quantum model)
- `mnist_quantum_nn.pth`: Trained model weights
- `notebooks/`: Optional Jupyter versions of the project (Gonna be messy - will be using this for Quantum ML experimenting)

## Run this project

From the terminal:

```bash
pip install torchquantum
python mnist_nn.py
```
