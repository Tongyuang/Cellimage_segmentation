from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import sys


class TxtLoader(Dataset):
    def __init__(self, txt_path, transform_all, transform_data):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1]))# words0:initial image, words1: label
        self.imgs = imgs 
        self.transform_all = transform_all
        self.transform_data = transform_data

    def __getitem__(self, index):
        fn, lb = self.imgs[index]
        # transform all
        img = np.asarray(Image.open(fn)).reshape((628, 628, 1))
        label = np.asarray(Image.open(lb)).reshape((628, 628, 1))
        comb = np.concatenate([img, label, label], axis=2)
        comb = Image.fromarray(comb.astype('uint8')).convert('RGB')
        comb = self.transform_all(comb)
        comb = np.asarray(comb)
        img = Image.fromarray(comb[:,:, 0]) # Image, L
        label = np.array(comb[:,:, 1])  # np.array, Len*Len
        label[label > 0] = 1# 0-1

        # transform data
        img = self.transform_data(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
