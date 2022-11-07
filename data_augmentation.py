import pandas as pd
from PIL import Image
from itertools import product, permutations
from collections import Counter
from torchvision import transforms
from tqdm import tqdm
import re
import random

'''
Augmentation Methods Used
1. image augmentation(Conventional)
2. sentence split
3. cross pair augmentation
'''

def cross_shuffle(overview, imdir):
    return [*map(list, zip(*product(overview, imdir)))]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomPerspective(),
    transforms.RandomResizedCrop((224,224))
])

def augment_img(img_paths, num, start_idx):
    img_l = len(img_paths)
    oimg_paths, nimg_paths = [], []
    for i in range(max(num, img_l)):
        img_p = f"./image/aug/aug{start_idx}.jpg"
        
        img = Image.open('./open'+img_paths[i%img_l][1:])
        img = transform(img) if i>=img_l else img
        img.save('./open'+img_p[1:])
        
        if i<img_l: oimg_paths.append(img_p)
        if i<num: nimg_paths.append(img_p)
        start_idx+=1
        
    return oimg_paths, nimg_paths, start_idx


def augment_text(texts, num):
    ntexts, text_l = [], []
    a = num**(1/3)
    for t in texts:
        t.replace('\n','')
        text_l+=[i for i in re.split('(<br \/>|<br>|\. )', t) if i and i!='. ' and i!='<br>' and i!='<br />']
        if len(text_l)>a: break
    
    for i in range(3):
        ntexts+=[*map('.'.join,permutations(text_l, 3-i))]
    
    random.shuffle(ntexts)
    return ntexts[:num]
        
        
    

if __name__=='__main__':

    df = pd.read_csv('./open/train.csv', sep=',', encoding='utf-8')

    classes = list(set(df.cat3))
    cat3 = Counter(df.cat3.tolist())
    M = cat3.most_common(1)[0][1]
    
    start_index=0
    imgs, txts, cat3s = [], [], []
    for c in tqdm(classes):
        cdf = df[df.cat3==c]
        rM = int((M-len(cdf))**0.5)
        
        img, txt = [], []
        oimg, img, start_index = augment_img(cdf.img_path.tolist(), rM, start_index)
        if rM:
            txt = augment_text(cdf.overview.tolist(), rM)
            img, txt = cross_shuffle(img, txt)
        else: img = []
        
        img, txt = oimg+img, cdf.overview.tolist()+txt
        if len(img)!=len(txt):print(len(img)==len(txt))
        imgs, txts, cat3s = imgs+img, txts+txt, cat3s+[c]*len(img)
        
    print(*map(len, (imgs, txts, cat3s)))
    
    aug_data = pd.DataFrame({
        'img_path': imgs,
        'overview': txts,
        'cat3': cat3s
    })
    
    aug_data.to_csv('./open/aug.csv', sep='\t')