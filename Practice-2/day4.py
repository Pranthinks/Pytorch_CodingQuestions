import torch
import torch.nn as nn

#Question 17
class Simple_Forward(nn.Module):
    def __init__(self, input, hid, output):
        super().__init__()
        self.fc1 = nn.Linear(input, hid)
        self.fc2 = nn.Linear(hid, output)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.rand(4, 4, dtype = torch.int).to(device)

model = Simple_Forward(4, 5, 6)
model.eval()

#Convert to TorchScript with torch.jit.trace
model_jit = torch.jit.trace(model, x)
model_jit.save("model_jit.pt")
#We can use htis model
op = model_jit(x)

#Convert to TorchScript with torch.jit.script
model_script = torch.jit.script(model)
model_script.save("model_script.pt")
op1 = model_script(x)

#Question - 18

#using torch.compile to compile the model in inference

model_comp = torch.compile(model)
op2 = model_comp(x)

#Question - 19
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
    record_shapes=True,
    with_stack=True
) as prof:
    with torch.no_grad():
        with record_function("model_inference"):
            output = model(x)

#Question 20
from torch.cuda.amp import autocast, GradScaler
import torch
model.eval()

with torch.cuda.amp.autocast():
    output = model(x)

#Question 21
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear},
    dtype = torch.qint8
)

#Question 22
class Sample_LLM(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed = nn.Linear(embed, embed)
        
    def forward(self, x):
       x = self.embed(x)
       return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.rand(4, 4, dtype = torch.float32).to(device)
model = Sample_LLM(4)

wei = model.embed.weight
mask = wei.abs() < 0.1

wei.data[mask] = 0.0 

#Question 23 and 24
from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
batch = tokenizer.vocab_size()
model.eval()
bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id

prompt = "I love the transformer"

def generate(prompt, max_words: int = 40, temperature: float = 0.8, vocab_size = batch):

    inputs = tokenizer.encode(prompt, add_special_tokens = False)

    input_1 = [bos_id] + inputs if bos_id is not None else inputs[:]

    input_tensor = torch.tensor(input_1, dtype = torch.long, pin_memory= True).to(device)
    generated_ids = []
    
    for _ in range(max_words):
        with torch.no_grad():
            output = model(input_tensor)

            logits = output.logits

            output1 = logits[:, -1, :] / temperature

            next_val = torch.argmax(output1, dim = -1).item()


            if next_val == eos_id and eos_id is not None:
                break
            generated_ids.append(next_val)

            op_tensor = torch.tensor([[next_val]], dtype = torch.long, pin_memory = True).to(device)

            input_tensor = torch.cat([input_tensor, op_tensor], dim = 1)
    res = tokenizer.decode(generated_ids, skip_special_tokens = True)

    return res
generate(prompt, 40, 0.8, batch)






