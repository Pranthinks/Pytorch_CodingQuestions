import torch
import torch.nn as nn
import math

#x = torch.tensor([1.0, 2.3, 3.9], dtype = torch.float32)
#y = torch.tensor([9.0, 3.5, 6.7], dtype = torch.float32)

#Question 3 Implementing own Softmax
class Custom_Softmax:
    def __init__(self):
        pass
    def Prob(self, x):
        exp_x = torch.exp(x - torch.max(x))
        probs = exp_x / torch.sum(exp_x)
        return probs
    
'''
obj = Custom_Softmax()
op = obj.Prob(y)
print('This is the final tensor:', op)
print(torch.argmax(op))
'''

#Question 4 

class Simple_Forward(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x

'''
x = torch.rand(3, 4, dtype = torch.float32)
y = torch.rand(3, 6, dtype = torch.float32)

model = Simple_Forward(4, 5, 6)

op = model(x)
print(op)
'''

#Qurstion 5

tokens = torch.tensor([1.2, 3.0, 4.1], dtype = torch.long)
class Custom_Embed(nn.Module):
    def __init__(self, embed_size , d_model):
        super().__init__()
        self.embed_layer = nn. Embedding(embed_size, d_model)
    
    def forward(self, x):
        x = self.embed_layer(x)
        return x
'''
op = Custom_Embed(5, 2)

print(op(tokens))
'''
#Question - 6

class Positional_Encoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        first_term = torch.exp((torch.arange(0, d_model, 2).float()) * (-math.log(10000.0)/ d_model))

        pe[: , 0::2] = torch.sin(position * first_term) 
        pe[:, 1::2] = torch.cos(position * first_term)

        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class Embed_and_pos(nn.Module):
    def __init__(self, embed_size, d_model, max_len):
        super().__init__()
        self.embed = nn.Embedding(embed_size, d_model)
        self.pos = Positional_Encoding(d_model, max_len)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.pos(x)
        return x

x = torch.tensor([[1, 2, 3, 4, 5]])

obj = Embed_and_pos(6, 4, 5)
print(obj(x))
