import torch
import torch.nn as nn
import torch.optim as optim


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LowRankLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Define learnable parameters A and B
        self.A = nn.Parameter(torch.randn(out_features, rank))
        self.B = nn.Parameter(torch.randn(rank, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Apply low-rank approximation to the weight matrix
        # (out_features, in_features)
        weight_approx = torch.matmul(self.A, self.B)
        return torch.matmul(x, weight_approx.T) + self.bias


# Example usage
batch_size = 32
input_features = 1024
output_features = 128
rank = 256

# Create an instance of the LowRankLinear layer
low_rank_layer = LowRankLinear(input_features, output_features, rank)

# Dummy input tensor (batch of inputs)
x = torch.randn(batch_size, input_features)

# Forward pass
output = low_rank_layer(x)

# Output should be (batch_size, output_features)
print("Shape of output:", output.shape)

# Define a simple neural network using LowRankLinear


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_features, 512)
        self.low_rank_layer = LowRankLinear(512, output_features, rank)
        # Assuming 10 classes for classification
        self.fc2 = nn.Linear(output_features, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.low_rank_layer(x)
        x = self.fc2(x)
        return x


# Create an instance of the network
model = SimpleNet()

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy input tensor (batch of inputs) and target tensor
x = torch.randn(batch_size, input_features)
target = torch.randint(0, 10, (batch_size,))
print(target)
# Training loop (simplified)
for epoch in range(100):  # number of epochs
    optimizer.zero_grad()   # zero the gradient buffers
    output = model(x)       # forward pass
    loss = criterion(output, target)  # compute the loss
    loss.backward()         # backward pass
    optimizer.step()        # update weights

    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
print(output)
print("Training complete.")
