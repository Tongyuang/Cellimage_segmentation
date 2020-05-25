from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import sys

class txtDataloader(Dataset):
    def __init__(self,txt_path,transform_3d,transform_1d):
        image_file = open(txt_path,'r')
        imgs = []
        for line in image_file:
            words = line.split()
            imgs.append((words[0],words[1]))
        self.imgs = imgs
        self.transform_3d = transform_3d
        self.transform_1d = transform_1d
        self.label_transformer = label_transformer
    def __getitem__(self,index):
        fn,lb = self.imgs[index]
        ini_img = np.asarray(Image.open(fn)).reshape(628,628,1)
        label = np.asarray(Image.open(lb)).reshape(628,628,1)
        comb = np.concatenate([ini_img,label,label],axis = 2)
        comb = Image.fromarray(comb.astype('uint8')).convert('RGB')
        comb = np.asarray(self.transform_3d(comb))
        img = Image.fromarray(comb[:,:,0])
        label = np.array(comb[:,:,1])
        label[label>0] = 1
        img = np.asarray(self.transform_1d(img))
        return img,label
    def __len__(self):
        return(len(self.imgs))
        
        