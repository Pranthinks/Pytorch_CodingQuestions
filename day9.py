import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataset, dataloader, random_split, TensorDataset
import math
#Transformers
#Question 26 Positional Encoding

class Postional_Encoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position_term = torch.arange(0, max_len, dtype = torch.float32).unsqueeze(1)
        div_term = torch.exp((-math.log(10000)/d_model)* torch.arange(0, d_model, 2).float())
        pe[:, 0::2] = torch.sin(position_term * div_term)
        pe[:, 1::2] = torch.cos(position_term * div_term)

        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1) ]
        return x
x = torch.zeros(5, 6, dtype= torch.float32)
obj = Postional_Encoding(5, 6)
op = obj(x)
print(op)
        