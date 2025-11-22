import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#Question 12 Implementing top-k sampling an imp step in LLM Inference

def tok_k(logits, k : int = 40):

    topk_logits, topk_indices = torch.topk(logits, k)

    probs = F.softmax(topk_logits, dim = -1)
    next_token = topk_indices[torch.multinomial(probs, 1)]

    return next_token.item()

#Question 13 Implementing top-p sampling

def top_p(logits, p: float = 0.8):
    #step -1 
    prob = F.softmax(logits, dim = -1)
    #Sort them in descending order
    sorted_probs , sorted_indices = torch.sort(prob, descending= True)
    #Calculate cummulative probabilites
    cum_probs = torch.cumsum(sorted_probs, dim= 0)
    mask = cum_probs <= p
    mask[0] = True

    filter_prb = sorted_probs[mask]
    filter_prb = filter_prb / filter_prb.sum()
    sampled_idx = torch.multinomial(filter_prb, 1)


    return sorted_indices[mask][sampled_idx]


#Question 14 Implementing KV - Cache 

class Custom_KV(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.atten = nn.MultiheadAttention(d_model, num_heads)
    

    def forward(self, x, cache = None):
        if cache is not None:
            past_key = cache["key"]
            past_val = cache["value"]
        
        else:
            past_key = None
            past_val = None
        
        key = torch.cat([past_key, x], dim = 0) if past_key is not None else x
        val = torch.cat([past_val, x], dim = 0) if past_val is not None else x

        atten_op, atten_wei = self.atten(x, key, val)

        new_val = {
            "key": key,
            "value" : val
        }

        return atten_op, new_val

obj = Custom_KV(4, 2)
x = torch.ones(2, 1, 4)
op1, op2 = obj(x, cache = None)
#print(op2.shape)
print(op2["key"].shape)

fin1 , fin2 = obj(x, cache = op2)
print(fin2["key"].shape)