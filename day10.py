import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader, TensorDataset, dataset
import math

#Question 29 Implementing the Transformer Encoder Block

class Feedforward_Network(nn.Module):
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
    
class Multihead(nn.Module):
    def __init__(self, num_heads, embeding):
        super().__init__()
        self.num_heads = num_heads
        self.embeding = embeding
        self.head = self.embeding // num_heads
        

        self.fc1 = nn.Linear(embeding, embeding)
        self.fc2 = nn.Linear(embeding, embeding)
        self.fc3 = nn.Linear(embeding, embeding)
        self.out_layer = nn.Linear(embeding, embeding)
    
    def forward(self, x):
        batch_size, seq_len , embed = x.size()
        Q = self.fc1(x)
        K = self.fc2(x)
        V = self.fc3(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head).transpose(1, 2)

        first_term = torch.matmul(Q, K.transpose(-2, -1))
        sec_term = math.sqrt(self.head)

        val = F.softmax(first_term/sec_term, dim=-1)

        output = torch.matmul(val, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed)

        out = self.out_layer(output)
        return out

'''
obj = Multihead(2, 4)
x = torch.rand(1, 4, 4)
op = obj(x)

feed = Feedforward_Network(4, 4, 4)
final = feed(op)
print(final)
'''
# Implementing Tranformer block(calling those two into this single block)

class Transformer_Encoder(nn.Module):
    def __init__(self, d_layer, heads, feed_hid):
        super().__init__()
        self.multi = Multihead(heads, d_layer)
        self.feed_for = Feedforward_Network(d_layer, feed_hid, d_layer)
        self.norm1 = nn.LayerNorm(d_layer)
        self.norm2 = nn.LayerNorm(d_layer)

    
    def forward(self, x):
        atten = self.multi(x)
        x = self.norm1(x + atten)

        fid = self.feed_for(x)
        x = self.norm2(x + fid)

        return x

obj = Transformer_Encoder(4 ,2, 4)
x = torch.rand(1, 3, 4)
output = obj(x)
print(output)
