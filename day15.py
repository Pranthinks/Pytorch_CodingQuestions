import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

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
        
        
        
tokens = ["I", "love", "machine", "learning"]
obj = fake_model(4, 2, token_labels=tokens)
x = torch.rand(1, 4, 4)
output = obj(x)
print(output)
