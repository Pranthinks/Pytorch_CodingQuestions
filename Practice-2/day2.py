import torch
import torch.nn as nn
import math

#x = torch.tensor([1.0, 2.3, 3.9], dtype = torch.float32)
#y = torch.tensor([9.0, 3.5, 6.7], dtype = torch.float32)

#Question 3 Implementing own Softmax
class Custom_Softmax:
    def __init__(self):
        pass
    def Prob(self, x):
        exp_x = torch.exp(x - torch.max(x))
        probs = exp_x / torch.sum(exp_x)
        return probs
    
'''
obj = Custom_Softmax()
op = obj.Prob(y)
print('This is the final tensor:', op)
print(torch.argmax(op))
'''

#Question 4 

class Simple_Forward(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x

'''
x = torch.rand(3, 4, dtype = torch.float32)
y = torch.rand(3, 6, dtype = torch.float32)

model = Simple_Forward(4, 5, 6)

op = model(x)
print(op)
'''

#Qurstion 5

tokens = torch.tensor([1.2, 3.0, 4.1], dtype = torch.long)
class Custom_Embed(nn.Module):
    def __init__(self, embed_size , d_model):
        super().__init__()
        self.embed_layer = nn. Embedding(embed_size, d_model)
    
    def forward(self, x):
        x = self.embed_layer(x)
        return x
'''
op = Custom_Embed(5, 2)

print(op(tokens))
'''
#Question - 6

class Positional_Encoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        first_term = torch.exp((torch.arange(0, d_model, 2).float()) * (-math.log(10000.0)/ d_model))

        pe[: , 0::2] = torch.sin(position * first_term) 
        pe[:, 1::2] = torch.cos(position * first_term)

        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class Embed_and_pos(nn.Module):
    def __init__(self, embed_size, d_model, max_len):
        super().__init__()
        self.embed = nn.Embedding(embed_size, d_model)
        self.pos = Positional_Encoding(d_model, max_len)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.pos(x)
        return x
'''
x = torch.tensor([[1, 2, 3, 4, 5]])
obj = Embed_and_pos(6, 4, 5)
print(obj(x))
'''

#Question 7 Implementing padding
from torch.nn.utils.rnn import pad_sequence

class Padding:
    def __init__(self):
        pass

    def start(self, val):
        output = pad_sequence(val, batch_first=True, padding_value=0)
        mask = (output == 0)
        return output, mask

x = [
    torch.tensor([1, 2, 3]),
    torch.tensor([1, 2, 5, 6, 5])
]

op = Padding()
print(op.start(x))

#Question 8 Writing a Basic inference loop to my custom text generation transformer

#Written in Google Colab first laoding the tokenizer and model
from google.colab import files
import zipfile
from transformers import PreTrainedTokenizerFast
import torch

val_token = files.upload()

val_first = list(val_token.keys())[0]

with zipfile.ZipFile(val_first, 'r') as zip_ref:
  zip_ref.extractall('./')

custom_tokenizer = PreTrainedTokenizerFast.from_pretrained("./tinystories_tokenizer")
vocab = custom_tokenizer.vocab_size
print(vocab)

def _bytelevel_detok(text: str) -> str:
    """Fallback cleanup for ByteLevel artifacts"""
    return text.replace("Ġ", " ").replace("Ċ", "\n").strip()

#Loading the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

val_model = files.upload()

val_sec = list(val_model.keys())[0]

def _bytelevel_detok(text: str) -> str:
    """Fallback cleanup for ByteLevel artifacts"""
    return text.replace("Ġ", " ").replace("Ċ", "\n").strip()

model = Chatbot(
    vocab_size=custom_tokenizer.vocab_size,
    heads=6,
    d_model=384,
    hid_layer=1536,
    seq_len=256,
    num_layers=4,
    dropout=0.1
).to(device)

ckpt = torch.load(val_sec, map_location = device)
missing , unexpected = model.load_state_dict(ckpt["model_state_dict"], strict = False)


#Inference Logic
bos_id = custom_tokenizer.bos_token_id
eos_id = custom_tokenizer.eos_token_id
pad_id = custom_tokenizer.pad_token_id


def generate(prompt : str, max_word_len : int = 20, temperature : float = 0.7):

  encoded_op = custom_tokenizer.encode(prompt, add_special_tokens = False)
  input = [bos_id] + encoded_op if bos_id is not None else encoded_op[:]

  matrix = torch.tensor(input, dtype = torch.long, device = device).unsqueeze(0)

  max_attr = getattr(model, "seq_len", 256)

  if matrix.size(1) > max_attr:
    matrix = matrix[:, -max_attr:]
  
  generated_ids = []

  for val in range(max_word_len):
    output = model(matrix)[:, -1 , :]

    output = output / temperature
    next_token = torch.argmax(output, dim = -1).item()

    if eos_id is not None and next_token == eos_id:
      break

    generated_ids.append(next_token)
    
    val_tensor = torch.tensor([[next_token]], dtype = torch.long, device = device)

    matrix = torch.cat([matrix, val_tensor], dim = 1)

    if matrix.size(1) > max_attr:
      matrix = matrix[:, -max_attr:]
  
  de_output = custom_tokenizer.decode(generated_ids, skip_special_tokens= True)
  de_output =   _bytelevel_detok(de_output)
  return de_output


prompt = "I love books"

print(generate(prompt = prompt, max_word_len = 15, temperature = 0.8))


#Question 9 Finding the size of the model
total_parameters = []
trainable_params = []
total_bytes = 0
for i in model.parameters():
  if i.requires_grad:
    trainable_params.append(i.numel())
  total_parameters.append(i.numel())

  total_bytes += i.numel()* i.element_size()



print("This is the total parameters the model has:", sum(total_parameters))
print("This is the total Trainable Parameters the model has:", sum(trainable_params))
print("Model Size in MB is:", total_bytes / (1024 ** 2))