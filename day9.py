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
'''
x = torch.zeros(5, 6, dtype= torch.float32)
obj = Postional_Encoding(5, 6)
op = obj(x)
print(op)
'''
    
#Question 27 Scaled Dot Product Attention Mechanism
import torch.nn.functional as F

def Manual_Attention(q_mat, k_mat, val_mat):
    d_k = q_mat.size(-1)

    first_term = torch.matmul(q_mat, k_mat.t())
    second_term = torch.sqrt(torch.tensor(d_k, dtype = torch.float32))

    val = first_term/second_term

    res = F.softmax(val, dim = -1)
    output = torch.matmul(res, val_mat)
    return output, res
'''
x = torch.rand(3, 4)
y = torch.rand(3, 4)
z = torch.rand(3, 2)
a, b = Manual_Attention(x, y, z)
print('The V matrix we got:', a)
print('Result after Softmax', b)
'''

#Question 28  Multi - Head Attention

class Multi_head_attention(nn.Module):
    def __init__(self, num_heads, embed_len):
        super().__init__()
        self.num_heads = num_heads
        self.embed_len = embed_len
        self.heads = self.embed_len//self.num_heads

        self.layer1 = nn.Linear(embed_len, embed_len)
        self.layer2 = nn.Linear(embed_len, embed_len)
        self.layer3 = nn.Linear(embed_len, embed_len)
        self.out_layer = nn.Linear(embed_len, embed_len)

    
    def forward(self, x):
        batch_size, seq_len, embedings = x.size()
        Q = self.layer1(x)
        K = self.layer2(x)
        V = self.layer3(x)

        Q= Q.view(batch_size, seq_len, self.num_heads, self.heads).transpose(1, 2)
        K= K.view(batch_size, seq_len, self.num_heads, self.heads).transpose(1, 2)
        V= V.view(batch_size, seq_len, self.num_heads, self.heads).transpose(1, 2)

        first = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(self.heads)
        val = F.softmax(first, dim = -1)

        output = torch.matmul(val, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embedings)

        out = self.out_layer(output)
        return out

obj = Multi_head_attention(2, 4)
x = torch.rand(1, 4, 4)
output = obj(x)
print(output)