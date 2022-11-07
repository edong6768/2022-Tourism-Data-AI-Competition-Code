from transformers.optimization import get_cosine_schedule_with_warmup

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from TourClassification import TourClassification1Aux as TourClassification
from TourClassification import TourDatasetAux

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def train(net, train_data, test_data, device):

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    
    epochs = 20
    loss = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=3e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps*0.1),
        num_training_steps=total_steps
    )

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_loss = 100
    best_state=None

    # net.load_state_dict(torch.load('./bestsaved-model.pt'))
    for epoch in range(epochs):
        acc = tot_loss = 0

        net.train()
        pbar = tqdm(train_loader)
        for img, text, label, label_aux1, label_aux2 in pbar:
            img, label, label_aux1, label_aux2, = map(lambda x: x.to(device), (img, label, label_aux1, label_aux2,))
            pred, pred_aux1, pred_aux2 = net(img, text)
            L = loss(pred, label)
            Lax1 = loss(pred_aux1, label_aux1)
            Lax2 = loss(pred_aux2, label_aux2)
            
            L_tot = L*0.85 + Lax1*0.05 + Lax2*0.1

            optimizer.zero_grad()
            L_tot.backward()
            optimizer.step()
            scheduler.step()
            
            tot_loss += L.cpu().detach().item()
            
            l_idx = torch.argmax(label.cpu(), dim=1)
            p_idx = torch.argmax(pred.cpu(), dim=1).detach()
            acc+=torch.sum(l_idx==p_idx).item()
            pbar.set_description(f'Epoch: {epoch+1}/{epochs}, correct:{acc}/{len(train_data)}, loss:{L.detach().item()}')
            
        acc/=len(train_data)
        tot_loss/=len(train_loader)
        print(tot_loss, best_loss)
        print(f'train acc:{acc*100}%, train_loss:{tot_loss}')
        
        
        train_losses.append(tot_loss)
        train_accs.append(acc)
        
        # test acc
        acc = tot_loss = 0
        net.eval()
        for img, text, label, _, _  in tqdm(test_loader):
            img, label, = map(lambda x: x.to(device), (img, label,))
            pred, _, _ = net(img, text)
            
            L = loss(pred, label)
            tot_loss += L.cpu().detach().item()
            
            l_idx = torch.argmax(label.cpu(), dim=1)
            p_idx = torch.argmax(pred.cpu(), dim=1).detach()
            acc+=torch.sum(l_idx==p_idx).item()
        acc/=len(test_data)
        tot_loss/=len(test_loader)
        print(f'test acc:{acc*100}%, test_loss:{tot_loss}')
        
        test_losses.append(tot_loss)
        test_accs.append(acc)

        if best_loss > tot_loss:
            best_state, best_loss = net.state_dict(), tot_loss
            torch.save(best_state, './bestsaved-model-auxver.pt')
            best_acc, best_epoch = acc, epoch
            print(f'Best_epoch: {best_epoch}, test acc:{best_acc*100}%, test_loss:{best_loss}')
    
    print(f'Best_epoch: {best_epoch}, test acc:{best_acc*100}%, test_loss:{best_loss}')

    plt.subplot(1, 2, 1)
    plt.plot(train_losses), plt.plot(test_losses)
    plt.subplot(1, 2, 2)
    plt.plot(train_accs), plt.plot(test_accs)
    plt.show()
    
    return best_state

def inference(net, classes, device):
    net.load_state_dict(torch.load('./bestsaved-model-auxver.pt'))
    
    test_data = TourDatasetAux(train=False)
    test_loader = DataLoader(test_data, batch_size=32)
    
    preds = []
    
    net.eval()
    for img, text in tqdm(test_loader):
        img, = map(lambda x: x.to(device), (img,))
        pred, _, _ = net(img, text)
        p_idx = torch.argmax(pred.cpu(), dim=1).detach()
        preds+=[classes[i.item()] for i in p_idx]
        
    df = pd.DataFrame({'id': test_data.df.id.tolist(), 'cat3': preds})
    df.to_csv('./open/submission.csv', sep=',', index=False)
    
    return preds


def ensemble_inference(net, states, classes, device):
     
    test_data = TourDatasetAux(train=False)
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

    train_data, test_data = TourDatasetAux.split_dataset(0.98)

    net = TourClassification(num_cls=train_data.num_cls,
                             num_cls_aux1=train_data.num_cls_aux1,
                             num_cls_aux2=train_data.num_cls_aux2,
                             device=device).to(device)

    train(net, train_data, test_data, device)
    preds = inference(net, train_data.classes, device)
    
    # states = ensemble_train(train_data.classes, device)
    # for i, s in enumerate(states):
    #     torch.save(s, f'./bestsaved-ensemble-model{i}.pt')
    # ensemble_inference(net, states, train_data.classes, device)


