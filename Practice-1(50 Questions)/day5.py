import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Question 15
class NeuralNetwork(nn.Module):
    def __init__(self, input_ch, hidden_ch, output_ch, dropout_prob = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_ch, hidden_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_ch, output_ch)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
'''
model = NeuralNetwork(2, 3, 5)
optimizer = optim.SGD(model.parameters(), lr= 0.01)
criterion = nn.MSELoss()
x = torch.rand(2, 2, dtype=torch.float32)
y = torch.rand(2, 5, dtype = torch.float32)
model.train()
for i in range(15):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print("This is Loss",loss)
model.eval()
with torch.no_grad():
    test_input = torch.rand(1, 2)
    prediction = model(test_input)
    print("This is Prediction", prediction)
'''
#Question 16
class BatchNormalization(nn.Module):
    def __init__(self, input_ch, hidden_ch, output_ch):
        super().__init__()
        self.fc1 = nn.Linear(input_ch, hidden_ch)
        self.batch = nn.BatchNorm1d(hidden_ch)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ch, output_ch)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.batch(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#Question 17 and 18
class SimpleNeuralNetwork(nn.Module):
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
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)
model = SimpleNeuralNetwork(4, 5, 8).to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.01)
criterion = nn.MSELoss()
x = torch.rand(3, 4, dtype = torch.float32)
y = torch.rand(3, 8, dtype = torch.float32)
model.train()
for i in range(15):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print("The loss in training", loss)
model.eval()
with torch.no_grad():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype = torch.float32)
    predic = model(x)
    print("The prediction is ", predic)
'''