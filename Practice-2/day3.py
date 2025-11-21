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


