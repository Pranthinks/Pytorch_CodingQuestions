import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.functional as F

#Question 1 and 2 and 4

class Simple_Model(nn.Module):
    def __init__(self, input, hid, output):
        super().__init__()
        self.fc1 = nn.Linear(input , hid)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid, output)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Simple_Model(4, 3, 6).to(device)
x = torch.rand(4, 4, dtype = torch.float32).to(device)
y = torch.rand(4, 6, dtype = torch.float32).to(device)
criterion = nn.MSELoss()

model.eval()
with torch.no_grad():
    for i in range(5):
        output = model(x)
        output_cpu = output.cpu()
        loss = criterion(output, y)

        print("This is the loss in Inference", loss)


