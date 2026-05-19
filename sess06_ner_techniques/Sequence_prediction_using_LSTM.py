import torch
import torch.nn as nn
import numpy as np

# -----------------------------
# 1. Prepare data
# -----------------------------
X = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7]
], dtype=np.float32)

y = np.array([4, 5, 6, 7, 8], dtype=np.float32)

# Convert to tensors
X = torch.tensor(X).unsqueeze(-1)   # shape: (samples, seq_len, 1)
y = torch.tensor(y).unsqueeze(-1)   # shape: (samples, 1)

# -----------------------------
# 2. Define LSTM model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # take last time step
        out = self.fc(out)
        return out

model = LSTMModel()

# -----------------------------
# 3. Loss & optimizer
# -----------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 4. Training
# -----------------------------
epochs = 100

for epoch in range(epochs):
    model.train()

    output = model(X)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -----------------------------
# 5. Prediction
# -----------------------------
model.eval()

test_input = torch.tensor([[6, 7, 8]], dtype=torch.float32).unsqueeze(-1)

with torch.no_grad():
    prediction = model(test_input)

print("Predicted value:", prediction.item())