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


    

