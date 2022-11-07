import torch
from torch import nn, optim
from TourClassification import TourClassification1, TourDataset, TourTrainer
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sd = 777
random.seed(sd)
torch.random.manual_seed(sd)

train_data, val_data = TourDataset.split_dataset(0.9)

net = TourClassification1(detach=False, device=device).to(device)

CFG = {
    'name': '',
    'train_data': train_data,
    'val_data': val_data,
    'batch_size': 32,
    'epochs': 5,
    'loss_func': nn.CrossEntropyLoss(),
    'lr': 3e-5,
    'optimizer': optim.Adam,
    'device': device
}

trainer = TourTrainer(net, **CFG)
trainer.train()
trainer.inference(TourDataset(cat3=train_data.cat3['cat'], train=False))

