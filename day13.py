import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import dataset, TensorDataset, dataloader

#Question 34 Simple Transformer Model

#Custom Positional Encoding  got from day9
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
# My Custom Postional Feed Forward Network taken from day12
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

# My Custom Masking FUnction taken from day12
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

# My Cstom Cross Encoder from day12
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



class FullTransformer_Custom(nn.Module):
    def __init__(self, vocab_size, heads, d_model, hidden_lay,seq_len, dropout = 0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.pos = Postional_Encoding(seq_len, d_model)
        
        #Encoder Stuff
        self.E_Multi = nn.MultiheadAttention(d_model, heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.E_FeedFor = Position_Feedforward(d_model, hidden_lay)
        self.norm2 = nn.LayerNorm(d_model)

        #Decoder stuf
        self.pos1 = Postional_Encoding(seq_len,d_model)
        self.D_Mask = Masked_Attention(heads, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.Cross_att = Multihead(heads, d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.D_Feedfor = Position_Feedforward(d_model, hidden_lay)
        self.norm5 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.out_layer = nn.Linear(d_model, vocab_size)

    
    def forward(self, src, tar):
        # Doing Encoder Stuff 
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos(x)

        x1, _ = self.E_Multi(x.transpose(0,1), x.transpose(0,1), x.transpose(0,1))
        x1 = x1.transpose(0,1)
        x = self.norm1(x + x1)

        x2 = self.E_FeedFor(x)
        enc_output = self.norm2(x + x2)

        # Doing the Decoder Stuff
        x = self.tgt_embedding(tar) * math.sqrt(self.d_model)
        x = self.pos1(x)
        x3 = self.D_Mask(x)
        x = self.norm3(x+ self.dropout(x3))

        
        x4 = self.Cross_att(x, enc_output)  
        x = self.norm4(x + self.dropout(x4))
        
        x5 = self.D_Feedfor(x)
        x = self.norm5(x + self.dropout(x5))
        
        output = self.out_layer(x)
        return output

#Usage 

obj = FullTransformer_Custom(10, 2, 4, 4, 4)
x = torch.randint(0,10, (2, 4))
y = torch.randint(0, 10, (2, 3))
output = obj(x, y)
print(output)
