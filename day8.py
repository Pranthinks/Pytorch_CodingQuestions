#Question 24 Finding Loss and Accuracy for validation data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

class SimpleNetwork(nn.Module):
    def __init__(self, input_ch, hidden_ch, output_ch):
        super().__init__()
        self.fc1 = nn.Linear(input_ch, hidden_ch)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ch, output_ch)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
x = torch.rand(30, 5, dtype = torch.float32)
y = torch.rand(30, 6, dtype = torch.float32)
model = SimpleNetwork(5, 4, 6)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)