import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from TourClassification import TourClassification1 as TourClassification
from TourClassification import TourDataset

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt



def train(net, train_data, test_data, device):

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    
    epochs = 5
    loss = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    losses = []
    best_loss = 100
    best_state=None

    # net.load_state_dict(torch.load('./ckpt/bestsaved-model.pt'))
    for epoch in range(epochs):
        acc = tot_loss = 0

        net.train()
        pbar = tqdm(train_loader)
        for img, text, label in pbar:
            img, label, = map(lambda x: x.to(device), (img, label,))
            pred = net(img, text)
            L = loss(pred, label)

            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            
            tot_loss += L.cpu().detach().item()
            
            l_idx = torch.argmax(label.cpu(), dim=1)
            p_idx = torch.argmax(pred.cpu(), dim=1).detach()
            acc+=torch.sum(l_idx==p_idx).item()
            pbar.set_description(f'Epoch: {epoch+1}/{epochs}, correct:{acc}/{len(train_data)}, loss:{L.detach().item()}')
            
        acc/=len(train_data)
        tot_loss/=len(train_loader)
        print(tot_loss, best_loss)
        print(f'train acc:{acc*100}%')
        
        losses.append(tot_loss)            
        
        # test acc
        acc = 0
        net.eval()
        for img, text, label in tqdm(test_loader):
            img, label, = map(lambda x: x.to(device), (img, label,))
            pred = net(img, text)
            l_idx = torch.argmax(label.cpu(), dim=1)
            p_idx = torch.argmax(pred.cpu(), dim=1).detach()
            acc+=torch.sum(l_idx==p_idx).item()
        acc/=len(test_data)
        print(f'test acc:{acc*100}%')
        
        if best_loss > tot_loss:
            best_state, best_loss = net.state_dict(), tot_loss
            torch.save(best_state, './ckpt/bestsaved-model.pt')
            best_acc, best_epoch = acc, epoch
    
    print(best_epoch, best_loss, best_acc)
        

    plt.plot(losses)
    plt.show()
    
    return best_state

def inference(net, classes, device):
    net.load_state_dict(torch.load('./ckpt/bestsaved-model.pt'))
    
    test_data = TourDataset(classes=classes, train=False)
    test_loader = DataLoader(test_data, batch_size=32)
    
    preds = []
    
    net.eval()
    for img, text in tqdm(test_loader):
        img, = map(lambda x: x.to(device), (img,))
        pred = net(img, text)
        p_idx = torch.argmax(pred.cpu(), dim=1).detach()
        preds+=[classes[i.item()] for i in p_idx]
        
    df = pd.DataFrame({'id': test_data.df.id.tolist(), 'cat3': preds})
    df.to_csv('./open/submission.csv', sep=',', index=False)
    
    return preds


def ensemble_inference(net, states, classes, device):
     
    test_data = TourDataset(classes=classes, train=False)
    test_loader = DataLoader(test_data, batch_size=32)
    
    def ensemble(img, text,  hard=False):
        logits = []
        for state in states:
            net.load_state_dict(state)
            pr = net(img, text)
            
            if hard:
                p_idx = torch.argmax(pr, dim=1)
                pr = F.one_hot(p_idx, num_classes=len(classes))
                
            logits.append(pr)
        
        pred = sum(logits)
        p_idx = torch.argmax(pred.cpu(), dim=1).detach()
        
        return [classes[i.item()] for i in p_idx]
    
    preds = []
    
    net.eval()
    for img, text in tqdm(test_loader):
        img, = map(lambda x: x.to(device), (img,))
        preds+=ensemble(img, text)
        
    df = pd.DataFrame({'id': test_data.df.id.tolist(), 'cat3': preds})
    df.to_csv('./open/submission.csv', sep=',', index=False)
    
    return preds
    

def ensemble_train(classes, device, num = 3):
    df = pd.read_csv('./open/train.csv', sep=',', encoding='utf-8')
    states = []
    for i in range(num):
        data = df.iloc[i*len(df)//3:(i+1)*len(df)//3]
        train_data, test_data = TourDataset.split_dataset(0.9, data, classes)
        net = TourClassification(device=device).to(device)

        states.append(train(net, train_data, test_data, device))
        
    return states
    

    
if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    train_data, test_data = TourDataset.split_dataset(0.9)

    net = TourClassification(device=device).to(device)

    train(net, train_data, test_data, device)
    preds = inference(net, train_data.classes, device)
    
    # states = ensemble_train(train_data.classes, device)
    # for i, s in enumerate(states):
    #     torch.save(s, f'./ckpt/bestsaved-ensemble-model{i}.pt')
    # ensemble_inference(net, states, train_data.classes, device)


