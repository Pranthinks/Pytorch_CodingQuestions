import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

#Question 42 Implementing KV - Cache
class KV_Cache(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.atten = nn.MultiheadAttention(d_model, heads)
    
    def forward(self, x, cache = None):
        if cache is not None:
            past_key = cache["key"]
            past_val = cache["value"]
        else:
            past_key = None
            past_val = None
        key = torch.cat([past_key, x], dim=0) if past_key is not None else x
        val = torch.cat([past_val, x], dim=0) if past_val is not None else x
        atten_op, atten_weight = self.atten(x, key, val)
        
        new_val = {
            "key": key,
            "value": val
        }
        return atten_op, new_val
# First token
model = KV_Cache(4, 2)
x1 = torch.rand(1, 1, 4)
op1, cache = model(x1, cache=None)
print("Output 1:", op1)
print("Cache after 1st token:", cache["key"].shape)

# Second token
x2 = torch.rand(1, 1, 4)
op2, cache = model(x2, cache=cache)
print("Output 2:", op2)
print("Cache after 2nd token:", cache["key"].shape)





#Question 43 Getting Visualizations for Attention weights

class fake_model(nn.Module):
    def __init__(self, d_model, heads, token_labels = None):
        super().__init__()
        self.token_labels = token_labels
        self.atten = nn.MultiheadAttention(d_model, heads, batch_first=True)
        
    def forward(self, x):
        attn_output, attn_weights = self.atten(x, x, x, need_weights=True, average_attn_weights=False)

        batch_size, n_heads, seq_len, _ = attn_weights.shape
        
        for head in range(n_heads):
            attn = attn_weights[0, head].cpu().detach().numpy()
            plt.figure(figsize=(6, 5))
            sns.heatmap(attn, annot=True, fmt = '.3f', cmap="viridis",
                    xticklabels=self.token_labels, yticklabels=self.token_labels)
            plt.xlabel("Key")
            plt.ylabel("Query")
            plt.title(f"Attention Head {head}")
            plt.show()
        return attn_output
        
        
'''
tokens = ["I", "love", "machine", "learning"]
obj = fake_model(4, 2, token_labels=tokens)
x = torch.rand(1, 4, 4)
output = obj(x)
print(output)
'''