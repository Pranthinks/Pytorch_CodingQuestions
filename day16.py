import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

#Question 47 Custom Pytorch Lightning Callback
class Lightning(Callback):
    def __init__(self, patience = 3):
        self.patience = patience
        self.val_loss = float('inf')
        self.wait = 0

    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print("🛑 Early stopping triggered!")
                trainer.should_stop = True

                
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

#Question 46 writing a training loop which has lightning integrated into it
class Lighting_version(pl.LightningModule):
    def __init__(self, input, hid, output):
        super().__init__()
        self.mod = Simple_Model(input, hid, output)
        self.loss = nn.MSELoss()
    
    def training_step(self, batch, batch_idx):
        x , y = batch
        y_pred = self.mod(x)
        loss = self.loss(y_pred, y)
        print("This is the Training loss", loss)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.mod(x)
        loss = self.loss(y_pred, y)
        print("This is the Validation Loss", loss)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr = 0.01)
    
model = Lighting_version(4,5, 6)
x = torch.rand(4, 4, dtype = torch.float32)
y = torch.rand(4, 6, dtype = torch.float32)

first = DataLoader(TensorDataset(x, y), batch_size=4)
second = DataLoader(TensorDataset(x, y), batch_size=4)
trainer = Trainer(max_epochs=20)
trainer.fit(model, first, second)