from .TourDataset import TourDataset
from .TourClassifiers import TourClassification1, TourClassificationTransformerAux
from .TourTrainer import AssessOnehot

from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class TourKFoldIter:
    _base_dir = './open/'
    
    def __init__(self, name, optimizer, epochs, batch_size=32, scheduler=None, lr=0.001, resize=224, fold=5, device=torch.device('cpu')):
        self.name = name
        self.fold = fold
        self.resize = resize
        self.batch_size = batch_size
        self.device = device
        
        self._data_preprocess()
        
        # self.nets = [TourClassification1(detach=False, device=device).to(device) for _ in range(fold)]
        self.nets = [TourClassificationTransformerAux(
                        num_cls=self.cat3['num_cls'],
                        num_cls_aux1=self.cat1['num_cls'],
                        num_cls_aux2=self.cat2['num_cls'],
                        detach=False,
                        device=device).to(device) for _ in range(fold)]
        self.optimizers = [optimizer(net.parameters(), lr) for net in self.nets]
        
        self.schedulers = []
        if scheduler:
            total_steps = len(self.train_loader) * epochs
            for opt in optimizer:
                self.schedulers.append(scheduler(
                                    opt,
                                    num_warmup_steps=int(total_steps*0.1),
                                    num_training_steps=total_steps
                                ))
        else: self.schedulers=[scheduler]*fold
        

    def _data_preprocess(self):
        df = pd.read_csv(self._base_dir + 'train.csv', sep=',', encoding='utf-8')
        self.df = self.stratifiedKFold(df, self.fold)
        
        self.cat3 = TourDataset._get_classes(df, 'cat3')
        self.cat1 = TourDataset._get_classes(df, 'cat1')
        self.cat2 = TourDataset._get_classes(df, 'cat2')
        
        train_transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomPerspective(),
                transforms.ToTensor()
            ])
        
        self.train_data = [TourDataset(df=self.df[self.df.kfold!=index], 
                                       cat3=self.cat3,
                                       transform=train_transform, 
                                       resize=self.resize) for index in range(self.fold)]
        self.val_data = [TourDataset(df=self.df[self.df.kfold==index], 
                                       cat3=self.cat3,
                                       resize=self.resize) for index in range(self.fold)]
        
        
        self.train_loader = [DataLoader(data, batch_size=self.batch_size, shuffle=True)
                             for data in self.train_data]
        self.val_loader = [DataLoader(data, batch_size=self.batch_size)
                             for data in self.val_data]
        
        self.assess_train = [AssessOnehot(f'train({i})', len(data), len(loader)) for i, data, loader in enumerate(zip(self.train_data, self.train_loader))]
        self.assess_val = [AssessOnehot(f'val({i})', len(data), len(loader)) for i, data, loader in enumerate(zip(self.val_data, self.val_loader))]
        
        self.best = []
        for _ in range(self.fold):
            self.best.append({
                'epoch': 0,
                'state': None,
                'acc': 0,
                'loss': 100,
            })
            
        
    def __getitem__(self, index):
        return [itr[index] for itr in (self.nets, self.optimizers, self.schedulers, self.train_loader, self.val_loader, self.assess_train, self.assess_val)]
    
    def pred_logit(self, image, text):
        logit = self.nets[0](image, text)
        for net in self.nets[1:]:
            pred = net(image, text)
            logit += net.pred_calc(pred)
        return logit
    
    def state_dict(self):
        return (net.state_dict() for net in self.nets)
    
    def save_all(self):
        for i, state_dict in enumerate(self.state_dict()):
            torch.save(state_dict, f'./ckpt/bestsaved-model{self.name}({i:02d}).pt')
    
    def save(self, index):
        torch.save(self.nets[index].state_dict, f'./ckpt/bestsaved-model{self.name}({index:02d}).pt')
    
    def autosave(self, index, epoch):
        if self.best[index]['loss'] > self.assess_val[index].get_data()['loss']:
            self.save(index)
            self.best[index] = {
                'epoch': epoch,
                'state': self.nets[index].state_dict(),
                **self.assess_val[index].get_data(),
            }
            print('New Best Model Saved')
    
    def load_state_dict(self):
        for i, net in enumerate(self.nets):
            net.load_state_dict(torch.load(f'./ckpt/bestsaved-model{self.name}({i:02d}).pt'))
    
    @classmethod
    def stratifiedKFold(cls, df, fold=5):
        folds = StratifiedKFold(n_splits=fold, random_state=42, shuffle=True)
        df['kfold'] = -1
        for i in range(fold):
            df_idx, valid_idx = list(folds.split(df.values, df['cat3']))[i]
            valid = df.iloc[valid_idx]

            df.loc[df[df.id.isin(valid.id) == True].index.to_list(), 'kfold'] = i

        return df


class TourEnsembleTrainer:
    def __init__(self, ensem, loss_func, device, name='', batch_size=32, epochs=5, lr=0.001):
        self.name = name
        self.ensem: TourKFoldIter =ensem
        
        self.device = device
        self.loss = loss_func.to(device)

        self.batch_size = batch_size
        self.epochs = epochs
        
        # self.train_losses, self.val_losses = [], []
        # self.train_accs, self.val_accs = [], []
        
        self.best ={
            'epoch': 0,
            'state': None,
            'acc': 0,
            'loss': 100,
        }
            
    def _validate(self, net, val_loader, assess_val):
        assess_val.reset()
        net.eval()
        for img, text, *labels in tqdm(val_loader):
            img, *labels, = map(lambda x: x.to(self.device), (img, *labels,))
            pred = net(img, text)
            
            ls = sum(tuple(net.loss_calc(self.loss, pred, labels)))/len(tuple(net.loss_calc(self.loss, pred, labels)))
            assess_val.accumulate(ls, net.pred_calc(pred), labels[0])
            
        print(self.assess_val)
        # self.val_losses.append(self.assess_val.get_data()['loss'])
        # self.val_accs.append(self.assess_val.get_data()['acc'])
        
           
    def _train_epoch(self, net, optimizer, scheduler, train_loader, assess_train, epoch):
        self.assess_train.reset()
        net.train()
        pbar = tqdm(train_loader)
        for img, text, *labels in pbar:
            img, *labels, = map(lambda x: x.to(self.device), (img, *labels,))
            pred = net(img, text)
            L = net.loss_calc(self.loss, pred, labels)
            
            optimizer.zero_grad()
            for l in L:
                l.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            ls = sum(tuple(net.loss_calc(self.loss, pred, labels)))/len(tuple(net.loss_calc(self.loss, pred, labels)))
            assess_train.accumulate(ls, net.pred_calc(pred), labels[0])
            pbar.set_description(f'Epoch: {epoch+1}/{self.epochs}, correct:{self.assess_train.acc}/{len(self.train_data)}, loss:{ls.item()}')
        
        print(assess_train)
        # self.train_losses.append(self.assess_train.get_data()['loss'])
        # self.train_accs.append(self.assess_train.get_data()['acc'])
        
        
    def train(self):
        for epoch in range(self.epochs):
            for i, (net, optimizer, scheduler, train_loader, val_loader, assess_train, assess_val) in enumerate(self.ensem):
                
                self._train_epoch(net, optimizer, scheduler, train_loader, assess_train, epoch)
                self._validate(net, val_loader, assess_val)
                self.ensem.autosave(i, epoch)
        
        # plt.subplot(1, 2, 1)
        # plt.plot(self.train_losses), plt.plot(self.val_losses)
        # plt.subplot(1, 2, 2)
        # plt.plot(self.train_accs), plt.plot(self.val_accs)
        # plt.show()
        
        # return self.best['state']
    
    def inference(self, test_data):
        classes = self.emsem.cat3['cat']
        self.ensem.load_state_dict()
        
        test_loader = DataLoader(test_data, batch_size=self.batch_size)
        
        preds = []
        
        self.net.eval()
        for img, text in tqdm(test_loader):
            img, = map(lambda x: x.to(self.device), (img,))
            pred = self.ensem.pred_logit(img, text)
            p_idx = torch.argmax(pred.cpu(), dim=1).detach()
            preds+=[classes[i.item()] for i in p_idx]
            
        df = pd.DataFrame({'id': test_data.df.id.tolist(), 'cat3': preds})
        df.to_csv('./open/submission.csv', sep=',', index=False)
        
        return preds