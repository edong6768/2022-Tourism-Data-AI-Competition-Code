import torch
from torch import nn, optim
from transformers.optimization import get_cosine_schedule_with_warmup
from TourClassification import TourClassificationTransformerAux, TourDatasetAux, TourTrainer
from TourClassification import FocalLoss
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sd = 777
random.seed(sd)
torch.random.manual_seed(sd)
if device=='cuda': torch.cuda.random.manual_seed(sd)

train_data, val_data = TourDatasetAux.split_dataset(0.9, resize=384)

net = TourClassificationTransformerAux(
    num_cls=train_data.cat3['num_cls'],
    num_cls_aux1=train_data.cat1['num_cls'],
    num_cls_aux2=train_data.cat2['num_cls'],
    detach=False,
    device=device).to(device)

CFG = {
    'name': 'TransformerAux',
    'train_data': train_data,
    'val_data': val_data,
    'batch_size': 16,
    'epochs': 20,
    'loss_func': FocalLoss(gamma=2),
    #'loss_func': nn.CrossEntropyLoss(),
    'lr': 3e-5,
    'optimizer': optim.AdamW,
    'scheduler': get_cosine_schedule_with_warmup,
    'device': device,
}

trainer = TourTrainer(net, **CFG)
trainer.train()
trainer.inference(TourDatasetAux(cat3=train_data.cat3['cat'], resize=384, train=False))

