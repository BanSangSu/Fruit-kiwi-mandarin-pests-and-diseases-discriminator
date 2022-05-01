# 실행 전 해야 할 것 console

train_acces = []
train_losses = []
val_accse = []
val_losses = []
best_loss = 1000000000000
one = 1
num = 0

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
# from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import StepLR

from Custom_CosineAnnealingWarmRestarts import CosineAnnealingWarmUpRestarts

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from PIL import Image
from PIL import ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# display images
from torchvision import utils
import matplotlib.pyplot as plt
# %matplotlib inline

# utils
import numpy as np
import time
import copy

# specify the data path
path2data = 'E:/kim/efg/dataset'

# if not exists the path, make the directory
if not os.path.exists(path2data):
    os.mkdir(path2data)
torch.multiprocessing.freeze_support()

# Data preprocessing
trans_train = transforms.Compose([transforms.Resize((984,1024)),
                            transforms.RandomCrop(900),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(degrees=(-90, 90)),
                            transforms.Resize((640,800)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
trans_val = transforms.Compose([transforms.Resize((640,800)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

trainset = torchvision.datasets.ImageFolder(root='E:/kim/efg/dataset/train',
                                            transform=trans_train)
validationset = torchvision.datasets.ImageFolder(root='E:/Kim/efg/dataset/val',
                                                 transform=trans_val)

class_name = ['정상','궤양병','귤응애','진딧물','점무늬병','총채벌레']
classes = trainset.classes


trainloader = DataLoader(trainset,
                         batch_size=4,
                         shuffle=True,
                         num_workers=6
                        )
valloader = DataLoader(validationset,
                       batch_size=1,
                       shuffle=False,
                       num_workers=3)

# # 데이터 배치당 보기
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# print(labels)

device_txt = 'cuda:0'
device = torch.device(device_txt if torch.cuda.is_available() else "cpu")

# load dataset
#train_ds = dataset.
from models import EfficientNet_b4
model_ft = EfficientNet_b4()

# Use when you want to load the state_dict
#model_ft.load_state_dict(torch.load('E:/kim/efg/model/efficientnet-b7_21-12-12/model_0_LOSS_0.8945174066314939.pth', map_location=device))
model_ft.to(device)
# for param in model_ft.parameters():
#     param.requires_grad = False


# optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_ft.parameters(), lr=1e-10, momentum=0.9, weight_decay=1e-4)
#scheduler
scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=60, T_mult=1, eta_max=1e-5, T_up=12, gamma=0.5)


#train
def fit(model, criterion, optimizer, epochs, train_loader, valid_loader):

    global num

    model.to(device)

    for epoch in range(epochs):

        if num > 600000:
            num = 0

        train_loss = 0
        train_num = 0
        now = time.time()

        for train_x, train_y in train_loader:

            num += 1

            model.train()
            train_x, train_y = train_x.to(device), train_y.to(device).long()

            optimizer.zero_grad()
            pred = model(train_x)
            loss = criterion(pred, train_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_num += 1

            if num % 1000 == 0:
                scheduler.step()

        train_avg_loss = train_loss / train_num
        print("Epoch : {}, Train_Avg_loss : {}".format(epoch, train_avg_loss))
        train_losses.append(train_avg_loss)

        # validation data check
        valid_loss = 0
        valid_num = 0

        for valid_x, valid_y in valid_loader:
            with torch.no_grad():
                model.eval()
                valid_x, valid_y = valid_x.to(device), valid_y.to(device).long()
                pred = model(valid_x)
                loss = criterion(pred, valid_y)

            valid_loss += loss.item()
            valid_num += 1
        val_avg_loss = valid_loss / valid_num
        print("Epoch : {}, Val_Avg_loss : {}".format(epoch, val_avg_loss))
        val_losses.append(val_avg_loss)

        plt.plot(val_losses, label='val_loss')
        plt.plot(train_losses, label='train_loss')
        plt.xlabel('efficientnet-b7_21-12-13')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

        # save path
        savepath = 'E:/kim/efg/model/efficientnet-b7_21-12-13/model_{}_LOSS_{}.pth'

        if best_loss > val_avg_loss:
            print("@@@@@@@@@@@ SAVE MODEL @@@@@@@@@@@")
            best_loss = val_avg_loss
            torch.save(model_ft.state_dict(), savepath.format(epoch, best_loss))

    print(time.time() - now)

def run():
    torch.multiprocessing.freeze_support()
    print("freeze")

if __name__ == '__main__':
    run()
    fit(model=model_ft, criterion=criterion,optimizer=optimizer,epochs=500, train_loader=trainloader, valid_loader=valloader)