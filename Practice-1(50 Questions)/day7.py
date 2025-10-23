import torch
import torch.nn as nn
import torch.optim as optim

class Simple(nn.Module):
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

#Question 22 Loss function switching between loses
class CustomLoss(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.type = type
        if self.type == 'L1':
            self.loss = nn.L1Loss()
        elif self.type == 'MSE':
            self.loss = nn.MSELoss()
    def forward(self, x, y):
        xi = self.loss(x, y)
        return xi
#Custom Loss Function
class Pran_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, y_pred, y):
        se = (y_pred-y)**2
        return se.mean()


'''
model = Simple(2, 3, 5)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = Pran_Loss()
criterion1 = CustomLoss('L1')
x = torch.rand([2, 2], dtype = torch.float32)
y = torch.rand([2, 5], dtype = torch.float32)

for i in range(20):
    optimizer.zero_grad()
    output = model(x)
    loss1 = criterion(output, y)
    loss2 = criterion1(output, y)
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()
    print('My custom loss function', loss1)
    print('My L1 Loss Function', loss2)

'''

#Question 23 Early Stopping
class EarlyStop():
    def __init__(self):
        self.max_val = float('inf')
    def check(self, loss):
        if loss > self.max_val:
            return True
        else:
            self.max_val = loss
model = Simple(2, 3, 5)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = Pran_Loss()
criterion1 = CustomLoss('L1')
x = torch.rand([2, 2], dtype = torch.float32)
y = torch.rand([2, 5], dtype = torch.float32)
obj = EarlyStop()

for i in range(20):
    optimizer.zero_grad()
    output = model(x)
    loss1 = criterion(output, y)
    
    if obj.check(loss1.item()):
        break
    loss1.backward()
    optimizer.step()
    print('My custom loss function', loss1)
    
         