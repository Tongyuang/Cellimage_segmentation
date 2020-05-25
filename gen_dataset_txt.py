import os
import numpy as np
train_text_name = 'train_dataset.txt'
valid_text_name = 'valid_dataset.txt'

train_dataset_path = 'dataset1/train'
train_SEG_path = 'dataset1/train_GT/SEG'




def gen_dataset_txt(train_dataset_path,train_SEG_path,train_text_name,valid_text_name,p=0.75):
    
    train_file = open(train_text_name,'w')
    valid_file = open(valid_text_name,'w')
    train_content = os.listdir(train_dataset_path)
    train_SEG_content = os.listdir(train_SEG_path)
    for i in range(len(train_content)):
        p0 = np.random.uniform(0,1)
        img_path = train_dataset_path+'/'+train_content[i]
        img_SEG_path = train_SEG_path+'/'+train_SEG_content[i]
        if(p0<0.75):
            train_file.write(img_path+' '+img_SEG_path)
            train_file.write('\n')
        else:
            valid_file.write(img_path+' '+img_SEG_path)
            valid_file.write('\n')

if __name__ == '__main__':
    gen_dataset_txt(train_dataset_path,train_SEG_path,train_text_name,valid_text_name,p=0.75)       