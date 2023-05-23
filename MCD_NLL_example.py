import torch
import torch.nn as nn
import torch.nn.functional as F

def nll_loss(preds, targets, predicted_variance):
  loss = 0.5 * 1/predicted_variance *(preds - targets)**2 + 0.5 * torch.log(predicted_variance)
  mean_loss = torch.mean(loss)

  return mean_loss

# Define the model
class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear_variance = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.dropout(F.relu(self.linear(x)))
        variance = self.softplus(self.linear_variance(x)) + 1e-6  # Predict variance
        return out, variance

# Initialize the model with the number of input and output features
model = LinearModel(input_size=10, output_size=1, dropout_rate=0.5)

# Define a loss function and an optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Example data
x = torch.randn(100, 10)  # 100 samples, 10 features each
y = torch.randn(100, 1)   # 100 target values

# Training loop
for epoch in range(10):  # 100 epochs
    model.train()
    y_pred, variance = model(x)  # Forward pass
    loss = nll_loss(y_pred, y, variance)  
    print(f"Epoch {epoch}, Loss {loss.item()}")

    optimizer.zero_grad()  # Zero gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

# Monte Carlo Dropout
model.train()
with torch.no_grad():
    predictions = torch.stack([model(x[:1])[0] for _ in range(100)])
    log_variances = torch.stack([model(x[:1])[1] for _ in range(100)])

mean = predictions.mean(dim=0)
variance_mcd = predictions.var(dim=0)
variance_nll = torch.exp(log_variances.mean(dim=0))

# Combine the variances
total_variance = variance_mcd + variance_nll

print(f"Prediction mean: {mean}, Total variance: {total_variance}")
print(variance_mcd, variance_nll)
