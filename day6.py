import torch 
import torch.nn as nn
import torch.optim as op
from torch.nn.utils import clip_grad_norm_

#Question 19 Learning Rate Scheduler and 20 Gradient clipping
class SimpleNetwork(nn.Module):
    def __init__(self, input_ch, hidden_ch, output_ch):
        super().__init__()
        self.fc1 = nn.Linear(input_ch, hidden_ch)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ch, output_ch)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
model = SimpleNetwork(2, 3, 5)
criterion = nn.MSELoss()
optimizer = op.SGD(model.parameters(), lr = 0.01)
scheduler = op.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)   # keeping lr scheduler
x = torch.rand(2, 2, dtype=torch.float32)
y = torch.rand(2, 5, dtype = torch.float32)
model.train()
for i in range(20):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)   # adding the gradient clipping which gonna normalize the gradients
    optimizer.step()
    scheduler.step()     # calling the lr scheduler
    print(f"Epoch {i+1}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.5f}")


#Question 21 Saving and loaded model state
#Saving the model
x = torch.rand(4, 4, dtype = torch.float32)
y = torch.rand(4, 4, dtype = torch.float32)
model = SimpleNetwork(4, 3, 4)
optimizer = op.SGD(model.parameters(), lr = 0.01)
scheduler = op.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  
criterion = nn.MSELoss()
for i in range(10):
    model.train()
    optimizer.zero_grad()
    op = model(x)
    loss = criterion(op, y)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)  
    optimizer.step()
    scheduler.step()
    print(loss)
torch.save({
    'epoch':20,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'val_loss':loss.item()
}, 'test_model.pth')
print('Model Saved Successfully')
#Loading the saved model
a = torch.load('test_model.pth')
model.load_state_dict(a['model'], strict=False)
print(f"✓ Model loaded from epoch {a['epoch']}")
print(f"✓ Best Val Loss: {a['val_loss']:.4f}\n")
