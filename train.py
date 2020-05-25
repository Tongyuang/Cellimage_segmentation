from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import TxtLoader
import Models
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def load_data(train_dir,valid_dir,input_size = 256, batch_size = 8):
    data_transforms = {
        'train_transform_3d':transforms.Compose([
            transforms.RandomCrop([300,300]),
            transforms.RandomHorizontalFlip(p = 0.25),
            transforms.RandomApply([transforms.RandomRotation(45)],p = 0.25),
            transforms.Resize([input_size,input_size])
        ]),
        'train_transform_1d':transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.5,0.5,0.5,0.5)],p=0.25),
            transforms.ToTensor()
        ]),
        'valid_transform_3d':transforms.Compose([
            transforms.Resize([input_size,input_size])
        ]),
        'valid_transform_1d':transforms.Compose([
            transforms.ToTensor()
        ])       
    }
    
    train_img = TxtLoader(train_dir,data_transforms['train_transform_3d'],data_transforms['train_transform_1d'])
    valid_img = TxtLoader(valid_dir,data_transforms['valid_transform_3d'],data_transforms['valid_transform_1d'])
    
    train_loader = DataLoader(train_img,batch_size=batch_size,shuffle=True)
    valid_loader = DataLoader(valid_img,batch_size=batch_size,shuffle=False)
    
    return train_loader,valid_loader

def train_model(device,model, train_loader, valid_loader, criterion, optimizer, num_epochs = 20):
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1)
    
    def train(device,model,train_loader,optimizer,criterion):
        model.train(True)
        total_loss = 0.0
        total_iou_num,total_iou_den = 0,0
        
        for img,label in tqdm(train_loader):
            inputs = img.to(device)
            label = label.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,label)
            _,prediction = torch.max(outputs,1)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item() * inputs.size(0)
            prediction,label = np.array(prediction.cpu()).reshape(-1)==1,np.array(label.cpu()).reshape(-1)==1
            
            total_iou_num += np.sum(prediction*label)
            total_iou_den += np.sum(prediction+label)
            
        epoch_loss = total_loss/len(train_loader.dataset)
        epoch_iou = total_iou_num/total_iou_den
        return epoch_loss,epoch_iou
    
    def valid(device,model,train_loader,criterion):
        model.train(False)
        total_loss = 0.0
        total_iou_num,total_iou_den = 0,0
        
        for img,label in tqdm(train_loader):
            inputs = img.to(device)
            label = label.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs,label)
            _,prediction = torch.max(outputs,1)
            
            total_loss += loss.item() * inputs.size(0)
            prediction,label = np.array(prediction.cpu()).reshape(-1)==1,np.array(label.cpu()).reshape(-1)==1
            
            total_iou_num += np.sum(prediction*label)
            total_iou_den += np.sum(prediction+label)
            
        epoch_loss = total_loss/len(train_loader.dataset)
        epoch_iou = total_iou_num/total_iou_den
        return epoch_loss,epoch_iou
    
    best_iou = 0
    for epoch in range(num_epochs):
        print('epoch:%d / %d'%(epoch,num_epochs))
        train_loss,train_iou = train(device,model,train_loader,optimizer,criterion)
        scheduler.step()
        print('Training:loss:%f,iou:%f'%(train_loss,train_iou))
        valid_loss,valid_iou = valid(device,model,train_loader,criterion)
        print('Validating:loss:%f,iou:%f'%(valid_loss,valid_iou))
        if(valid_iou>best_iou):
            best_iou = valid_iou
            best_model = model
            torch.save(best_model.state_dict(),'best_model.pt')
            
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

train_dir = './train_dataset.txt'
valid_dir = './valid_dataset.txt'
num_epochs = 100
lr = 0.001
model = Models.U_Net(in_ch=1,out_ch=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_loader,valid_loader = load_data(train_dir,valid_dir)
optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9)
criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1,1]).to(device))
train_model(device,model,train_loader,valid_loader,criterion,optimizer,num_epochs=num_epochs)
