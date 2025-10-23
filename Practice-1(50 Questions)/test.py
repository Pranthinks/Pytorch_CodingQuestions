import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

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
'''
import math
class Positional_Encoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        first_term = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        second_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model))  # (d_model/2,)
        pe[:, 0::2] = torch.sin(first_term*second_term)
        pe[:, 1::2] = torch.cos(first_term*second_term)

        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        x = x+ self.pe[:, :x.size(1) ]

        return x
'''
model = Positional_Encoding(10, 6)
x = torch.rand(1, 3, 6)
op = model(x)
print(op)
'''
class Attention(nn.Module):
    def __init__(self, d_model, embedings):
        super().__init__()
        self.d_model = d_model
        self.fc1 = nn.Linear(embedings, embedings)
        self.fc2 = nn.Linear(embedings, embedings)
        self.fc3 = nn.Linear(embedings, embedings)
        self.fc4 = nn.Linear(embedings, embedings)
    
    def forward(self, x):
        Q = self.fc1(x)
        K = self.fc2(x)
        V = self.fc3(x)

        first_term = torch.matmul(Q, K.transpose(-2, -1))
        second_term = math.sqrt(self.d_model)

        val = F.softmax(first_term/second_term, dim=-1)

        V = torch.matmul(val, V)

        op = self.fc4(V)
        return op
'''
model = Attention(4, 5)
x = torch.rand(4, 5)
op = model(x)
print(x)
'''
class Multihead_Attention(nn.Module):
    def __init__(self, num_heads, embedings):
        super().__init__()
        self.num_heads = num_heads
        self.embedings = embedings
        self.heads = self.embedings//self.num_heads
        self.fc1 = nn.Linear(embedings, embedings)
        self.fc2 = nn.Linear(embedings, embedings)
        self.fc3 = nn.Linear(embedings, embedings)
        self.fc4 = nn.Linear(embedings, embedings)
    
    def forward(self, x):
        batch_size, seq_len, embedings = x.size()
        Q = self.fc1(x)
        K = self.fc2(x)
        V = self.fc3(x)

        Q= Q.view(batch_size, seq_len, self.num_heads, self.heads).transpose(1, 2)
        K= K.view(batch_size, seq_len, self.num_heads, self.heads).transpose(1, 2)
        V= V.view(batch_size, seq_len, self.num_heads, self.heads).transpose(1, 2)

        first_term = torch.matmul(Q, K.transpose(-2, -1))
        second_term = math.sqrt(self.heads)

        val = F.softmax(first_term/second_term, dim=-1)

        output = torch.matmul(val, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embedings)

        op = self.fc4(output)
        return op
'''
model = Multihead_Attention(4, 8)
x = torch.rand(1, 3, 8)
op = model(x)
print(x)
'''

class Transformer_Encoder(nn.Module):
    def __init__(self, num_heads, embedings, hid_lay, dropout = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedings, hid_lay)
        self.Multi = Multihead_Attention(num_heads, embedings)
        self.norm = nn.LayerNorm(embedings)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedings)
        self.fc2 = nn.Linear(hid_lay, embedings)
    
    def forward(self, x):
        atten_op = self.Multi(x)

        x = self.norm(x + self.dropout(atten_op))

        x1= self.fc1(x)
        x2 = self.fc2(x1)
        x = self.norm1(x + self.dropout(x2))

        return x
         
'''
model = Transformer_Encoder(2, 8, 4)
x = torch.rand(1, 4, 8)
op = model(x)
print(op)
'''
class Custom_Layernorm(nn.Module):
    def __init__(self, d_model, eta = 0.1):
        super().__init__()
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eta = eta
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim = True)
        var = x.var(dim=-1, keepdim = True)
        first_term = x - mean
        second_term = torch.sqrt(var + self.eta)
        op = first_term/ second_term
        val = self.gamma*op + self.beta
        return val
'''
model = Custom_Layernorm(5)
x = torch.rand(3, 4)
op = model(x)
print(op)   
'''
class Padding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, x):
        op = pad_sequence(x, batch_first=True, padding_value = 0)

        mask = (op == 0)
        op = op.masked_fill(mask, float('-inf'))

        return op

model = Padding(4)
x = [torch.tensor([1.0, 2.0, 3.0]),
     torch.tensor([1.0, 3.0]),
     torch.tensor([1.0, 9.0, 45.9, 78.3])]
op = model(x)
print(op)