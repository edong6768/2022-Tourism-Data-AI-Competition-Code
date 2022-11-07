import torch
from torch import nn, optim
from transformers.optimization import get_cosine_schedule_with_warmup
from TourClassification import TourKFoldIter, TourEnsembleTrainer, TourDatasetAux
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sd = 777
random.seed(sd)
torch.random.manual_seed(sd)

kfoldCFG = {
    'name': 'KFold',
    'fold': 5,
    'epochs': 20, 
    'batch_size': 32,
    'lr': 3e-5,
    'optimizer': optim.AdamW,
    'scheduler': get_cosine_schedule_with_warmup,
    'device': device,
}
kfoldIter = TourKFoldIter(**kfoldCFG)


CFG = {
    'name': 'KFold',
    'batch_size': 32,
    'epochs': 20,
    'lr': 3e-5,
    'loss_func': nn.CrossEntropyLoss(),
    'device': device,
}

trainer = TourEnsembleTrainer(kfoldIter, **CFG)
trainer.train()
trainer.inference(TourDatasetAux(cat3=kfoldIter.cat3['cat3'], train=False))

