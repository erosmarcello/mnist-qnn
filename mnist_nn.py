# Eros Marcello | Quantum MNIST classifier using TorchQuantum.

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchquantum as tq
import torchquantum.functional as tqf
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # For visualization later

# Use GPU if available else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading ---
# Standard MNIST transform: ToTensor scales to [0, 1]
transform = transforms.ToTensor()
# Load datasets
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# DataLoaders. Batch 64 for training, shuffle enabled. Test batch 1 for single-image check later
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)

# --- Quantum Model Definition ---
class QuantumLayer(tq.QuantumModule):
    def __init__(self, n_wires: int, n_layers: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.cnot = tq.CNOT()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        self.q_device.reset_states(batch_size)

        # Encode classical features into quantum rotations.
        for wire in range(self.n_wires):
            tqf.ry(self.q_device, wires=wire, params=x[:, wire])

        # Variational quantum circuit.
        for _ in range(self.n_layers):
            for wire in range(self.n_wires):
                self.ry(self.q_device, wires=wire)
                self.rz(self.q_device, wires=wire)
            for wire in range(self.n_wires - 1):
                self.cnot(self.q_device, wires=[wire, wire + 1])

        return self.measure(self.q_device)


class QuantumMNIST(nn.Module):
    def __init__(self, n_wires: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, n_wires),
        )
        self.quantum = QuantumLayer(n_wires=n_wires, n_layers=n_layers)
        self.classifier = nn.Linear(n_wires, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_net(x)
        x = torch.tanh(x) * math.pi
        x = self.quantum(x)
        x = self.classifier(x)
        return x


# Instantiate model and move to target device
model = QuantumMNIST().to(device)

# --- Training Components ---
# CrossEntropyLoss for multi-class classification
criterion = nn.CrossEntropyLoss()
# Adam optimizer lr=0.001 is a decent default
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
epochs = 5  # of training passes
print("Training quantum model...")

model.train()  # Set model to training
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:  # Iterate over batches
        images, labels = images.to(device), labels.to(device)  # Data to device

        optimizer.zero_grad()  # Reset gradients
        output = model(images)  # Forward pass
        loss = criterion(output, labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()  # Accumulate batch loss.

    avg_loss = total_loss / len(train_loader)  # Average loss for the epoch.
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")  # Log progress.

print("Finished training.")

# --- Save Model State ---
# Save only the model weights (state_dict). Efficient.
torch.save(model.state_dict(), "mnist_quantum_nn.pth")
print("Saved model state_dict to mnist_quantum_nn.pth")

# --- Quick Inference Test ---
# Verify on a single test image.
data_iter = iter(test_loader)  # Get iterator.
test_image, test_label = next(data_iter)  # Get first sample.

# Set model to evaluation mode. Important for layers like Dropout/BatchNorm if used.
model.eval()
test_image = test_image.to(device)  # Image to device.
with torch.no_grad():  # Disable gradient calculation for inference.
    output = model(test_image)  # Get model output
    predicted_label = torch.argmax(output).item()  # Get predicted class index

print(f"Predicted label: {predicted_label}, Actual label: {test_label.item()}")

# Display the test image and result
plt.imshow(test_image.cpu().squeeze(), cmap="gray")  # Plot image (CPU, remove batch dim)
plt.title(f"Prediction: {predicted_label} | Actual: {test_label.item()}")  # Add title
plt.axis("off")  # Clean axes
plt.show()

# --- Reload Model Check ---
# Optional step here but this verifies loading works
loaded_model = QuantumMNIST().to(device)  # Create a new model instance.
loaded_model.load_state_dict(torch.load("mnist_quantum_nn.pth", map_location=device))  # Load weights
loaded_model.eval()  # Set to eval
print("✅ Model reloaded — all keys matched successfully.")
