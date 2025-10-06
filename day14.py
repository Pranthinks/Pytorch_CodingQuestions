import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader, TensorDataset, dataset
from torch.nn.utils.rnn import pad_sequence
import math
import torch.optim as optim

#Question 35  Doing Padding and Padding-Mask 
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 2.0, 3.0, 5.0, 6.0])
z = torch.tensor([1.0, 2.0])

output = pad_sequence([x, y, z], batch_first = True, padding_value=0)
#print(output)
#print(output.shape)
mask = (output==0).unsqueeze(1).unsqueeze(2)
#print(mask)
scores = torch.rand(3, 1, 1, 5)
#print(scores)
op = scores.masked_fill(mask, float('-inf'))
#print(op)

#Question 36 Custom Activation functions
#Creating ReLU
class Custom_Relu(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min = 0)

#Creating Custom GeLU
class Custom_GeLu(nn.Module):
    def forward(self, x):
        val = x * 0.5 * (1 + torch.erf(x/math.sqrt(2)))
        return val

#Writing Simple Feed Forward Network to test that ReLU
class Feed_Forward(nn.Module):
    def __init__(self, input_ch, hid_ch, output_ch):
        super().__init__()
        self.fc1 = nn.Linear(input_ch, hid_ch)
        self.gelu = Custom_GeLu()
        self.fc2 = nn.Linear(hid_ch, output_ch)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)

        return x

#Question 37 Performing Gradient Accumulation in the training loop
x = torch.rand(4, 3, dtype = torch.float32)
y = torch.rand(4, 3, dtype = torch.float32)
model = Feed_Forward(3, 4, 3)
op = optim.SGD(model.parameters() , lr=0.01)
criterion = nn.MSELoss()
grad_accum = 4

for i in range(20):
    model.train()
    op.zero_grad()
    for j in range(grad_accum):
        output = model(x)
        loss = criterion(output, y)
        loss = loss/grad_accum
        loss.backward()
    if (j+1)%grad_accum == 0:
        op.step()
        op.zero_grad()
    print(loss)


#Question 38 Using PyTorch’s AMP for mixed-precision training
model1 = Feed_Forward(3, 4, 5).to('cuda')
y = torch.rand(3, 3, device ='cuda', dtype = torch.float32)
op = torch.rand(3, 5, device = 'cuda', dtype = torch.float32)
criterion = nn.MSELoss()
optimizer = optim.SGD(model1.parameters(), lr=0.01)

scaler = torch.cuda.amp.GradScaler()

for i in range(20):
    model1.train()
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        output = model1(y)
        loss = criterion(op, output)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print(loss)

