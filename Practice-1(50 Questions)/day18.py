import torch
import torch.nn as nn
import torch.optim as optim


class Simple_Model(nn.Module):
    def __init__(self, input, hid, output):
        super().__init__()
        self.fc1 = nn.Linear(input, hid)
        self.fc2 = nn.Linear(hid, output)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
#Question 49 Profiling GPU Memory Usage
def log_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 **2
        print('Allocated memory is ', allocated)
        print('Reserved memorey is ', reversed)

    else:
        print("You dont have any GPU in your system")




#Using that GPU prfile in training and Inference


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = Simple_Model(4, 5, 6).to(device)
x = torch.rand(4, 4, dtype = torch.float32).to(device)
y = torch.rand(4, 6, dtype = torch.float32).to(device)
criterion = nn.MSELoss()
op = optim.SGD(model.parameters(), lr = 0.01)

model.train()
for i in range(5):
    op.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    log_memory()
    print('This is the loss in Training', loss)
    op.step()

model.eval()
with torch.no_grad():
    for i in range(5):
        output = model(x)
        loss = criterion(output , y)
        log_memory()
        print('This is the loss in Inference', loss)

