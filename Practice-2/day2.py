import torch
import torch.nn as nn


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

x = torch.rand(3, 4, dtype = torch.float32)
y = torch.rand(3, 6, dtype = torch.float32)

model = Simple_Forward(4, 5, 6)

op = model(x)
print(op)