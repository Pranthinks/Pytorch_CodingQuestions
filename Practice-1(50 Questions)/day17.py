import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

#Question 48 Implementing Distributed Data Parallel
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

def train(rank, world_size):
    #Initializing the Process Group
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank
    )


    torch.cuda.set_device(rank)
    
    model = Simple_Model().cuda(rank)
    model = DDP(model, device_ids=[rank])
    x = torch.rand(4, 4, dtype = torch.float32)
    y = torch.rand(4, 6, dtype = torch.float32)
    criterion = nn.MSELoss()
    op = optim.SGD(model.parameters(), lr = 0.01)
    for i in range(20):
        op.zero_grad()
        y_out = model(x)
        loss = criterion(y_out, y)

        loss.backward()
        op.step()
        if rank == 0:
            print("The loss is ", loss.item())
    
    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()







