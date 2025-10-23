import torch
import torch.nn as nn
import torch.optim as optim

#question 12
class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
'''
model = FeedForward(2, 3, 5)
x = torch.tensor([[2.0, 3.0]], dtype = torch.float32)
y = torch.tensor([[1.0, 1.0, 2.0, 5.0, 6.0]], dtype = torch.float32)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for i in range(20):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(y, output)
    loss.backward()
    optimizer.step()

    print(loss)

'''
#Question 13

class FeedforwardRelu(nn.Module):
    def __init__(self, input_d, hidden_d, output_d, activation = "ReLU"):
        super().__init__()
        self.fc1 = nn.Linear(input_d, hidden_d)
        self.fc2 = nn.Linear(hidden_d, output_d)
        if activation.lower() == "relu":
            self.rel = nn.ReLU()
        elif activation.lower() == "gelu":
            self.rel = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.rel(x)
        x = self.fc2(x)
        return x
'''
model = FeedforwardRelu(4, 6, 6, activation = 'ReLU')
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype = torch.float32)
y = torch.tensor([[2.0, 3.0, 6.0, 8.0, 10.0, 4.0]], dtype = torch.float32)
for i in range(20):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(loss)
'''

#Question 14
#Writing CNN Layers
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_chan, output_chan, kernal_size, stride = 1,padding = 0):
        super().__init__()
        self.input_chan = input_chan
        self.output_chan = output_chan
        self.kernal_size = kernal_size if isinstance(kernal_size, tuple) else (kernal_size, kernal_size)
        self.stride = stride
        self.padding = padding
        nn.Parameter()
        self.weight = nn.Parameter(torch.randn(output_chan, input_chan, *self.kernal_size)*0.01)
        self.bias = nn.Parameter(torch.randn(output_chan))
    
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

model = CNN(3, 16, 3)
x = torch.randn(1, 3, 32, 32)
y = model(x)
print(y.shape)