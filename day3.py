import torch
import torch.nn as nn
import torch.optim as optim
#Question 11
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, input):
        return self.linear(input)

x = torch.tensor([[2.0, 3.0]], dtype = torch.float32)
y = torch.tensor([[1.0, 3.0, 4.0]], dtype = torch.float32)
model = LinearRegression(input_dim=2, output_dim=3)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
for i in range(20):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(loss)

