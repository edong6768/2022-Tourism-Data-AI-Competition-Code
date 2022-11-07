import torch
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt



'''
CFG = {
    'name': name
    'train_data': train_data,
    'val_data': val_data,
    'batch_size': batch_size,
    'epochs': epochs,
    'loss_func': loss_func,    
    'lr': lr,
    'optimizer': optimizer,
    'scheduler': scheduler,
    'device': device,
}
'''


class AssessOnehot:
    def __init__(self, name, acc_length, loss_length):
        self.acc_length = acc_length
        self.loss_length = loss_length
        self.name = name
        self.reset()
    
    def reset(self):
        self.acc = self.tot_loss = 0
        self.curr_acc_len = self.curr_loss_len = 0
        
    def accumulate(self, loss, pred, label):
        self.tot_loss += loss.cpu().detach().item()
        
        l_idx = torch.argmax(label.cpu(), dim=1)
        p_idx = torch.argmax(pred.cpu(), dim=1).detach()
        self.acc+=torch.sum(l_idx==p_idx).item()
        
        self.curr_acc_len+=l_idx.detach().shape[0]
        self.curr_loss_len+=1
    
    def get_data(self):
        return {'acc': self.acc/self.curr_acc_len, 'loss': self.tot_loss/self.curr_loss_len}
        
    def __str__(self):
        return f'{self.name} acc:{self.acc/self.acc_length*100}%, {self.name} loss:{self.tot_loss/self.loss_length}'
        

class TourTrainer:
    def __init__(self, net, train_data, val_data, loss_func, optimizer, device, name='', scheduler=None, batch_size=32, epochs=5, lr=0.001):
        self.name = name
        self.net = net
        self.train_data, self.val_data = train_data, val_data
        
        self.loss = loss_func.to(device)
        
        self.device = device
        
        self.batch_size = batch_size
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size)

        self.epochs = epochs
        
        self.optimizer = optimizer(net.parameters(), lr=lr)

        if scheduler:
            total_steps = len(self.train_loader) * epochs
            self.scheduler = scheduler(
                                self.optimizer,
                                num_warmup_steps=int(total_steps*0.1),
                                num_training_steps=total_steps
                            )
        else: self.scheduler=scheduler
        
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []
        
        self.assess_train = AssessOnehot('train', len(train_data), len(self.train_loader))
        self.assess_val = AssessOnehot('val', len(val_data), len(self.val_loader))
        
        self.best ={
            'epoch': 0,
            'state': None,
            'acc': 0,
            'loss': 100,
        }
    
    def _autosave(self, epoch):
        if self.best['loss'] > self.assess_val.get_data()['loss']:
            torch.save(self.net.state_dict(), f'./ckpt/bestsaved-model{self.name}.pt')
            self.best = {
                'epoch': epoch,
                'state': self.net.state_dict(),
                **self.assess_val.get_data(),
            }
            print('New Best Model Saved')
    
    def _validate(self):
        self.assess_val.reset()
        self.net.eval()
        for img, text, *labels in tqdm(self.val_loader):
            img, *labels, = map(lambda x: x.to(self.device), (img, *labels,))
            pred = self.net(img, text)
            
            ls = sum(tuple(self.net.loss_calc(self.loss, pred, labels)))/len(tuple(self.net.loss_calc(self.loss, pred, labels)))
            self.assess_val.accumulate(ls, self.net.pred_calc(pred), labels[0])
            
        print(self.assess_val)
        self.val_losses.append(self.assess_val.get_data()['loss'])
        self.val_accs.append(self.assess_val.get_data()['acc'])
        
    
    def _train_epoch(self, epoch):
        self.assess_train.reset()
        self.net.train()
        pbar = tqdm(self.train_loader)
        for img, text, *labels in pbar:
            img, *labels, = map(lambda x: x.to(self.device), (img, *labels,))
            pred = self.net(img, text)
            L = self.net.loss_calc(self.loss, pred, labels)
            
            self.optimizer.zero_grad()
            for l in L:
                l.backward()
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            ls = sum(tuple(self.net.loss_calc(self.loss, pred, labels)))/len(tuple(self.net.loss_calc(self.loss, pred, labels)))
            self.assess_train.accumulate(ls, self.net.pred_calc(pred), labels[0])
            pbar.set_description(f'Epoch: {epoch+1}/{self.epochs}, correct:{self.assess_train.acc}/{len(self.train_data)}, loss:{ls.item()}')
        
        print(self.assess_train)
        self.train_losses.append(self.assess_train.get_data()['loss'])
        self.train_accs.append(self.assess_train.get_data()['acc'])
        
        
    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            self._validate()
            self._autosave(epoch)
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses), plt.plot(self.val_losses)
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs), plt.plot(self.val_accs)
        plt.show()
        
        return self.best['state']

    def inference(self, test_data):
        classes = self.train_data.cat3['cat']
        self.net.load_state_dict(torch.load(f'./ckpt/bestsaved-model{self.name}.pt'))
        
        test_loader = DataLoader(test_data, batch_size=self.batch_size)
        
        preds = []
        
        self.net.eval()
        for img, text in tqdm(test_loader):
            img, = map(lambda x: x.to(self.device), (img,))
            pred = self.net(img, text)
            pred = self.net.pred_calc(pred)
            p_idx = torch.argmax(pred.cpu(), dim=1).detach()
            preds+=[classes[i.item()] for i in p_idx]
            
        df = pd.DataFrame({'id': test_data.df.id.tolist(), 'cat3': preds})
        df.to_csv('./open/submission.csv', sep=',', index=False)
        
        return preds
