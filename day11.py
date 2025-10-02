import torch
import torch.nn as nn
from torch.utils.data import dataloader, dataset, TensorDataset
import torch.nn.functional as F

#Question 31 Postion wise Feed-Forward Network

class Position_Feedforward(nn.Module):
    def __init__(self, input_ch, output_ch, dropout = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_ch, output_ch)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_ch, input_ch)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
         x= self.fc1(x)
         x = self.relu(x)
         x = self.dropout(x)
         x = self.fc2(x)

         return x

'''
obj = Position_Feedforward(4, 6)
x = torch.rand(2, 4, dtype = torch.float32)
output = obj(x)
print(output)
'''

#Question 32 Layer Normalization from scratch

class LayerNorm1(nn.Module):
    def __init__(self, input, eps = 1e-6):
        super().__init__()
        self.input = input
        self.gamma = nn.Parameter(torch.ones(input))
        self.beta = nn.Parameter(torch.zeros(input))
        self.eps = eps
    
    def forward(self, x):

        mean = x.mean(dim=-1, keepdim = True)
        var = x.var(dim= -1, keepdim = True)

        val = (x - mean)/torch.sqrt(var + self.eps)

        output = self.gamma*val + self.beta

        return output

'''
obj = LayerNorm1(4)
x = torch.rand(4, 4)
output = obj(x)
print(output)
'''

#Question 33 Custom Transformer Decoder Layer


        