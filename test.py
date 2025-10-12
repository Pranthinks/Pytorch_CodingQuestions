import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

'''
x = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]]])
print(x)
y = x[:, 2:, 2:]
print(y)

a = torch.ones(1, 3)
b = torch.zeros(4, 3)
print(a)
print(b)
print(a+b)
a = torch.rand(2, 5, dtype = torch.float32)
b = a.to(torch.int64)
print(b)

x = torch.tensor([1.0, 2.3], requires_grad = True)
y = torch.rand(2)
f = x ** 2
f.backward(y)
print(x.grad)
'''
class simple_ff(nn.Module):
    def __init__(self, inn, out):
        super().__init__()
        self.fc1 = nn.Linear(inn, out)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        return x
'''
model = simple_ff(4, 4)
x = torch.rand(4, 4)

with torch.no_grad():
    op = model(x)
    print(op)

with torch.inference_mode():
    op1= model(x)
    print(op1)
'''
class Custom_AutoGrad(torch.autograd.Function):

    @staticmethod    
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2
    
    @staticmethod
    def backward(ctx, x):
        val, = ctx.saved_tensors
        grad_op = val * 2*x
        return grad_op
        
'''
x = torch.tensor([3.0], requires_grad=True)
y = Custom_AutoGrad.apply(x)
# Step 3: Compute gradients
y.backward()
print(y)
print(x.grad)
'''
#From Question 11 to 20 
class Custom_Relu(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.clamp(x, min = 0)

class Twolayer_MlP(nn.Module):
    def __init__(self, inp, hid, op, activ = 'ReLu', Dropout = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(inp, hid)
        if activ.lower() == 'custom':
            self.relu = Custom_Relu()
        elif activ.lower() == 'relu':
            self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid, op)
        self.dropout = nn.Dropout(Dropout)
        self.norm = nn.BatchNorm1d(hid)

    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = Twolayer_MlP(4, 3, 5, activ='Custom').to(device)
x = torch.rand(4, 4, dtype = torch.float32).to(device)
y = torch.rand(4, 5, dtype = torch.float32).to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  
criterion = nn.MSELoss()
for i in range(10):
    model.train()
    optimizer.zero_grad()
    op = model(x)
    loss = criterion(op, y)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)  
    optimizer.step()
    scheduler.step()
    print(loss)
'''
#Saving the model
x = torch.rand(4, 4, dtype = torch.float32)
y = torch.rand(4, 4, dtype = torch.float32)
model = Twolayer_MlP(4, 3, 4)
optimizer = optim.SGD(model.parameters(), lr = 0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  
criterion = nn.MSELoss()
for i in range(10):
    model.train()
    optimizer.zero_grad()
    op = model(x)
    loss = criterion(op, y)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)  
    optimizer.step()
    scheduler.step()
    print(loss)
torch.save({
    'epoch':20,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'val_loss':loss.item()
}, 'test_model.pth')
print('Model Saved Successfully')
#Loading the saved model
a = torch.load('test_model.pth')
model.load_state_dict(a['model'], strict=False)
print(f"✓ Model loaded from epoch {a['epoch']}")
print(f"✓ Best Val Loss: {a['val_loss']:.4f}\n")

