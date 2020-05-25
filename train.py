from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import TxtLoader
import Models as models
import os
import matplotlib.pyplot as plt
import numpy as np

## Note that: here we provide a basic solution for loading data and transforming data.
## You can directly change it if you find something wrong or not good enough.

## the mean and standard variance of imagenet dataset
## mean_vals = [0.485, 0.456, 0.406]
## std_vals = [0.229, 0.224, 0.225]

def load_data(data_dir="./dataset1/train", input_size=256, batch_size=8):
    data_transforms = {
        'train_3d': transforms.Compose([
            transforms.RandomCrop([300, 300]),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomApply([transforms.RandomRotation(45)], p=0.3),
            transforms.Resize([input_size,input_size]),
        ]),
        'train_1d': transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)], p=0.3),
            transforms.ToTensor()
        ]),
        'valid_3d': transforms.Compose([
            transforms.Resize([input_size,input_size]),
        ]),
        'valid_1d': transforms.Compose([
            transforms.ToTensor()
        ])
    }

    image_dataset_train = TxtLoader('./train_dataset.txt', data_transforms['train_3d'], data_transforms['train_1d'])
    image_dataset_valid = TxtLoader('./valid_dataset.txt', data_transforms['valid_3d'], data_transforms['valid_1d'])


    train_loader = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(image_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader


def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=20):

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        global lr
        lr = lr / (1 + (epoch // 5) * 2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(model, train_loader, optimizer, criterion):
        model.train(True)
        total_loss = 0.0
        total_in, total_un = 0, 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs.shape)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)# one-hot转为普通编码
            #print(predictions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            predictions, labels = np.array(predictions.cpu()).reshape(-1)==1, np.array(labels.cpu()).reshape(-1)==1
            
            cell_in = np.sum(predictions * labels)
            cell_un = np.sum(predictions + labels)

            total_in += cell_in
            total_un += cell_un

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_iou = total_in / total_un
        return epoch_loss, epoch_iou

    def valid(model, valid_loader, criterion):
        model.train(False)
        total_loss = 0.0
        total_in, total_un = 0, 0

        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)

            total_loss += loss.item() * inputs.size(0)
            predictions, labels = np.array(predictions.cpu()).reshape(-1)==1, np.array(labels.cpu()).reshape(-1)==1
            
            cell_in = np.sum(predictions * labels)
            cell_un = np.sum(predictions + labels)

            total_in += cell_in
            total_un += cell_un
            
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_iou = total_in / total_un
        return epoch_loss, epoch_iou

    best_iou = 0.0
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch)
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 10)
        train_loss, train_iou = train(model, train_loader, optimizer, criterion)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_iou))
        valid_loss, valid_iou = valid(model, valid_loader, criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_iou))
        if valid_iou > best_iou:
            best_iou = valid_iou
            best_model = model
            torch.save(best_model.state_dict(), 'best_model.pt')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    ## about training
    num_epochs = 100
    lr = 0.001

    ## model initialization
    model = models.U_Net(in_ch=1, out_ch=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = load_data()
    '''
    all_0, all_1 = 0, 0
    for inputs, labels in train_loader:
        all_0 += np.sum((labels==0).numpy())
        all_1 += np.sum((labels==1).numpy())
    for inputs, labels in valid_loader:
        all_0 += np.sum((labels==0).numpy())
        all_1 += np.sum((labels==1).numpy())
    print(all_0, all_1)
    sys.exit(0)
    '''

    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    ## loss function
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1]).to(device))
    train_model(model,train_loader, valid_loader, criterion, optimizer, num_epochs=num_epochs)
