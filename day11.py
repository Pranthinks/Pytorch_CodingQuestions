import torch
import torch.nn as nn
from torch.utils.data import dataloader, dataset, TensorDataset
import torch.nn.functional as F
import math
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

#Custom Multihead attention calculation
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
    
    def forward(self, x, enc_output=None):
        batch_size, seq_len, embed = x.size()
        Q = self.fc1(x)
        if enc_output is not None:
            K = self.fc2(enc_output)
            V = self.fc3(enc_output)
            seq_len_kv = enc_output.size(1)
        else:
            K = self.fc2(x)
            V = self.fc3(x)
            seq_len_kv = seq_len

        # Split heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head).transpose(1, 2)
        K = K.view(batch_size, seq_len_kv, self.num_heads, self.head).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.num_heads, self.head).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        # Combine heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed)
        return self.out_layer(out)
    
#Custom Masked Attention Calculation
class Masked_Attention(nn.Module):
    def __init__(self, num_heads, embedings):
        super().__init__()
        assert embedings % num_heads == 0
        self.num_heads = num_heads
        self.embedings = embedings
        self.heads = embedings // num_heads  

        self.fc1 = nn.Linear(embedings, embedings)
        self.fc2 = nn.Linear(embedings, embedings)
        self.fc3 = nn.Linear(embedings, embedings)
        self.outlayer = nn.Linear(embedings, embedings)
    
    def forward(self, x):
        batch_size, seq_len , embed = x.size()
        Q = self.fc1(x)
        K = self.fc2(x)
        V = self.fc3(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.heads).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.heads).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.heads).transpose(1, 2)
        
        first_term = torch.matmul(Q, K.transpose(-2, -1))
        second_term = math.sqrt(self.heads)
        scores = first_term/second_term
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask.to(scores.device), float('-inf'))

        val = F.softmax(scores, dim=-1)

        output = torch.matmul(val, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed)

        out = self.outlayer(output)
        return out


#Question 33 Custom Transformer Decoder Layer thats uses all the customly built methods

class Transformer_Decoder(nn.Module):
    def __init__(self, heads, d_model, hid, dropout = 0.1):
        super().__init__()
        self.masked = Masked_Attention(heads, d_model)
        self.Cross_Atten = Multihead(heads, d_model)
        self.Pos_Forward = Position_Feedforward(d_model, hid, dropout)
        self.norm1 = LayerNorm1(d_model)
        self.norm2 = LayerNorm1(d_model)
        self.norm3 = LayerNorm1(d_model)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x, enc_output = None):
        x2 = self.masked(x)
        x = self.norm1(x + self.dropout(x2))

        if enc_output is not None:
            x2 = self.Cross_Atten(x, enc_output)  
            x = self.norm2(x + self.dropout(x2))
        
        x2 = self.Pos_Forward(x)
        x = self.norm3(x + self.dropout(x2))

        return x

# Calling the Transformer decoder
obj = Transformer_Decoder(2, 4, 4)
x = torch.rand(1, 3, 4)
output = obj(x)
print(output)
        

        