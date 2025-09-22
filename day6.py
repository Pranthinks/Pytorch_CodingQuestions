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
def model_load(model, path = 'model.pth' ):

    torch.save(model.state_dict(), path)
    print('Model Saved Successfully')

def load_model(model_class, path = 'model.pth',  *model_args, **model_kwargs):
    model = model_class(*model_args, **model_kwargs)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model

model = SimpleNetwork(2, 4, 6)
model_load(model, 'first.pth')
loaded_model = load_model(SimpleNetwork, 'first.pth', 2, 4, 6)
x = torch.rand(1, 2)
print(loaded_model(x))

