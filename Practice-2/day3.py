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
