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
'''
#print(op2.shape)
print(op2["key"].shape)

fin1 , fin2 = obj(x, cache = op2)
print(fin2["key"].shape)
'''

#Question 15
from torch.nn.utils.rnn import pad_sequence

x = [ torch.tensor([1, 2, 3, 10, 4, 6, 7, 8,], dtype=torch.int),
      torch.tensor([1, 5, 6, 8], dtype=torch.int), 
      torch.tensor([1, 9, 10, 13, 18], dtype=torch.int), 
      torch.tensor([6, 8], dtype=torch.int) ]

sorted_x = sorted(x, key= lambda x: len(x))

def batch(batch_len : int , seq: list):
    i = 0
    while i < len(seq):
        matrix = seq[i: i + batch_len]
        val_pad = pad_sequence(matrix, batch_first= True, padding_value= 0)
        mask = (val_pad == 0)
        print(mask)
        i= i+batch_len

batch(2, sorted_x)

#Question - 16
#Micro Batching Can handle multiple inputs in the same time

def micro_generate(prompt : list[str], max_words : int = 40, vocab = vocab_size, temperature : float = 1.0, micro_batch : int = 2):
  input_list = []
  for i in range(len(prompt)):
    input = tokenizer.encode(prompt[i], add_special_tokens = False)
    input_id = [bos_id] + input if bos_id is not None else input[:]
    input_list.append(input_id)
  max_val = getattr(model, 'max_seq_len', 256)
  final_output = []
  
  for i in range(0, len(input_list), micro_batch):

    batch_val = input_list[i : i + micro_batch]
    
    #Finding the max_len of each batch_val and doing the padding to all based on this
    max_len = max(len(seq) for seq in batch_val)
    padded_batch = [
        seq + [pad_id] * (max_len - len(seq)) for seq in batch_val
    ]
    split_tensor = torch.tensor(padded_batch, dtype = torch.long).to(device)

    if split_tensor.size(1) > max_val:
      split_tensor = split_tensor[:, -max_val:]
    generated_ids = [[] for _ in range(len(split_tensor))]

    for _ in range(max_words):
      with torch.no_grad():
        output = model(split_tensor)[:, -1, :]
        output = output / temperature

        next_token = torch.argmax(output, dim = -1)

        for j , token in enumerate(next_token):
         if token.item() == eos_id:
           continue
         generated_ids[j].append(token.item())

        split_tensor = torch.cat([split_tensor, next_token.unsqueeze(1)], dim = 1)

        if split_tensor.size(1) > max_val:
          split_tensor = split_tensor[:, -max_val:]

    decoded_outputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
    final_output.extend(decoded_outputs)
     

  
     

