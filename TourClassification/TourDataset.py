from PIL import Image
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class TourDataset(Dataset):
    _base_dir = './open/'
    _train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomPerspective(),
        transforms.ToTensor()
    ])
    
    def __init__(self, df=None, cat3=None, transform=None, resize=224, train=True):
        super(TourDataset, self).__init__()
        self.load_data(df, cat3, train)
        
        self.resize = transforms.Resize((resize, resize))
        self.transform = transforms.ToTensor() if transform is None else transform
    
    
    def load_data(self, df=None, cat=None, train=True):
        self.train=train
        data = 'train' if train else 'test'
        
        self.df = pd.read_csv(self._base_dir + f'{data}.csv', sep=',', encoding='utf-8') if df is None else df
        
        if train:
            cat = self.get_classes(df, 'cat3') if cat is None else cat
            self.cat3 = self._labels(cat)
              
    
    @classmethod
    def _get_classes(cls, df, column):
        return sorted(list(set(df[column])))
        
    @classmethod
    def _labels(cls, cat):
        num_cls = len(cat)
        class_dict = dict(zip(cat, range(num_cls)))
        return {'cat': cat, 'num_cls': num_cls, 'class_dict': class_dict}
        
    def _get_onehots(self, data):
        label = F.one_hot(torch.tensor(self.cat3['class_dict'][data.cat3]), self.cat3['num_cls']).float()
        return label,
       
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        data = self.df.iloc[index]
        
        image = Image.open(self._base_dir + data['img_path'][1:])
        image = self.resize(image)
        image = self.transform(image)/255
        text = data.overview
        
        if self.train: 
            return image, text, *self._get_onehots(data)
        else:
            return image, text
    
    @classmethod
    def _stratified_split(cls, df, cat3, split):
        train_df, val_df = pd.DataFrame(), pd.DataFrame()
        for c in cat3:
            curr_df = df[df.cat3==c]
            split_idx = round(len(curr_df)*split)
            
            train_df = pd.concat([train_df, curr_df.iloc[:split_idx]], ignore_index=True)
            val_df = pd.concat([val_df, curr_df.iloc[split_idx:]], ignore_index=True)
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
        
        
    @classmethod
    def split_dataset(cls, split=0.8, df=None, cat3=None, resize=224):
        df = pd.read_csv(cls._base_dir + 'train.csv', sep=',', encoding='utf-8') if df is None else df
        
        cat3 = cls._get_classes(df, 'cat3') if cat3 is None else cat3
        train_df, val_df = cls._stratified_split(df, cat3, split)
        
        return cls(df=train_df, cat3=cat3, transform=cls._train_transform, resize=resize), cls(df=val_df, cat3=cat3, resize=resize)

    
    
class TourDatasetAux(TourDataset):
    def __init__(self, df=None, cat3=None, cat1=None, cat2=None, transform=None, resize=224, train=True):
        super(TourDatasetAux, self).__init__(df=df, cat3=cat3, transform=transform, resize=resize, train=train)
        if train:
            cat = self.get_cat(df, 'cat1') if cat1 is None else cat1
            self.cat1 = self._labels(cat)

            cat = self.get_cat(df, 'cat2') if cat2 is None else cat2
            self.cat2 = self._labels(cat)
            
    
    def _get_onehots(self, data):
        label = F.one_hot(torch.tensor(self.cat3['class_dict'][data.cat3]), self.cat3['num_cls']).float()
        label_aux1 = F.one_hot(torch.tensor(self.cat1['class_dict'][data.cat1]), self.cat1['num_cls']).float()
        label_aux2 = F.one_hot(torch.tensor(self.cat2['class_dict'][data.cat2]), self.cat2['num_cls']).float()
        return label, label_aux1, label_aux2
    
    
    @classmethod
    def split_dataset(cls, split=0.8, df=None, cat3=None, cat1=None, cat2=None, resize=224):
        df = pd.read_csv(cls._base_dir + 'train.csv', sep=',', encoding='utf-8') if df is None else df
        
        cat = dict()
        for i, c in enumerate((cat1, cat2, cat3)):
            cat[f'cat{i+1}'] = cls._get_classes(df, f'cat{i+1}') if c is None else c

        train_df, val_df = cls._stratified_split(df, cat['cat3'], split)
        
        return cls(df=train_df, **cat, transform=cls._train_transform, resize=resize), cls(df=val_df, **cat, resize=resize)

