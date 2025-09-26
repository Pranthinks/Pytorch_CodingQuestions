#Question 24 Finding Loss and Accuracy for validation data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split

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


def validate(model, criterion, val):
    model.eval()
    with torch.no_grad():
        val_inputs = torch.stack([dataset[i][0] for i in val])
        val_target = torch.stack([dataset[i][1] for i in val])
        output = model(val_inputs)
        loss = criterion(output, val_target)

        print(' The Loss we got on validation data is', loss)
        accur = (1-loss)*100
        print('The accuracy we got is', accur,'%')


'''
x = torch.rand(55, 5, dtype = torch.float32)
y = torch.rand(55, 6, dtype = torch.float32)
model = SimpleNetwork(5, 4, 6)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)
dataset = TensorDataset(x, y)
train , test , val = random_split(dataset, [30, 10, 15]) 
trai = train.indices
va = val.indices
for i in range(23):
    model.train()
    train_inputs = torch.stack([dataset[i][0] for i in trai])
    train_targets = torch.stack([dataset[i][1] for i in trai])
    optimizer.zero_grad()
    output = model(train_inputs)
    loss = criterion(output, train_targets)
    loss.backward()
    optimizer.step()
    validate(model= model, criterion=criterion, val = va)
'''

#Question 25 Performing Data Augmentation and creating custom data loader
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms
from PIL import Image
import os

class CustomDataloader(Dataset):
    def __init__(self, image_path, transform):
        self.transform = transform
        self.image_path = image_path
        self.image_files = os.listdir(image_path)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_path, self.image_files[index])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image

#Data Augmentation
transform = transforms.Compose([
    transforms.Resize((65, 65)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

dataset = CustomDataloader('files', transform= transform)
dataloader = DataLoader(dataset, batch_size=1)

for batch in dataloader:
    print(batch.shape)  