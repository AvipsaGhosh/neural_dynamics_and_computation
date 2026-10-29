import torch
import torch.nn as nn
import torch.optim as optim

# 1. Generate a training dataset: single gratings at random orientations and contrasts
np.random.seed(42)
torch.manual_seed(42)

n_samples = 600
train_images = []
train_targets = []

for _ in range(n_samples):
    ori = np.random.uniform(0, np.pi)
    cnt = np.random.uniform(0.1, 1.0)
    stim = create_grating(theta=ori, contrast=cnt)
    
    # Ground truth response comes from the biophysical divisive normalization model
    resp = divisive_normalization_model(stim, r_max=r_max_est, sigma_c=sigma_val)
    
    train_images.append(stim)
    train_targets.append(resp)

X_train = torch.tensor(np.array(train_images), dtype=torch.float32).unsqueeze(1) # Shape: (N, 1, 31, 31)
y_train = torch.tensor(np.array(train_targets), dtype=torch.float32).unsqueeze(1) # Shape: (N, 1)

# 2. Define a standard backprop CNN architecture
class V1BackpropModel(nn.Module):
    def __init__(self):
        super(V1BackpropModel, self).__init__()
        # Layer 1: Learns spatial convolutional filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=15, padding=7)
        self.relu = nn.ReLU()
        # Layer 2: Pool & project to a single neural response
        self.fc = nn.Linear(8 * 31 * 31, 1)
        
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc(out))  # Firing rates are non-negative
        return out

cnn_model = V1BackpropModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.003)

# 3. Train via backpropagation
epochs = 80
batch_size = 32

for epoch in range(epochs):
    permutation = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        predictions = cnn_model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

print(f"Training finished. Final Loss: {loss.item():.4f}")
