import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint, Callback


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



