import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import dataset, TensorDataset, dataloader

# My Custom Postional Feed Forward Network
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

# My Custom Masking FUnction
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

# My Cstom Cross Encoder
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




#Question 34 Implementing Full Transformer Architecture

class Full_Transformer(nn.Module):
    def __init__(self, heads, d_model, hid, dropout = 0.1):
        super().__init__()
        # Encoder Stuff
        self.E_Multi = Multihead(heads, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.E_Feed = Position_Feedforward(d_model, hid)
        self.norm2 = nn.LayerNorm(d_model)

        # Decoder Stuff
        self.D_Mask = Masked_Attention(heads, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.Cros_atten = Multihead(heads, d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.Pos_Feed = Position_Feedforward(d_model, hid)
        self.norm5 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, enc_output = None):
        # Doing Encoder Stuff 
        x1 = self.E_Multi(x)
        x = self.norm1(x + x1)

        x2 = self.E_Feed(x)
        x = self.norm2(x + x2)

        # Doing the Decoder Stuff
        x3 = self.D_Mask(x)
        x = self.norm3(x+ self.dropout(x3))

        if enc_output is not None:
            x4 = self.Cros_atten(x, enc_output)  
            x = self.norm4(x + self.dropout(x4))
        
        x5 = self.Pos_Feed(x)
        x = self.norm5(x + self.dropout(x5))

        return x

obj = Full_Transformer(2, 4, 4)
x = torch.rand(1, 5, 4)
output = obj(x)
print(x)






