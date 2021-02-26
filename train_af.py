#!/usr/bin/env python

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, utils
import torchvision.transforms.functional as TF

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value) 
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars\n
        torch.backends.cudnn.deterministic = True  #needed\n
        torch.backends.cudnn.benchmark = False

seed = 42
random_seed(seed,True)

batch_size = 128
learning_rate = 3e-4
num_epochs = 100
num_workers = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

df = pd.read_csv('path') #emr data를 사용

features_reduction = [
        
]# emr data에 존재하는 환자 feature
labels = ['hx_af_new']

age_max = df.loc[:,'age'].max()
age_min = df.loc[:,'age'].min()
ini_max = df.loc[:,'ini_nih'].max()
ini_min = df.loc[:,'ini_nih'].min()
print(df.describe(), '\n')

for j in range(df.shape[0]):
        value = df.loc[j,'age']
        new_value = (value - age_min) / (age_max - age_min)
        df.loc[j, 'age'] = new_value
        value = df.loc[j, 'ini_nih']
        new_value = (value - ini_min) / (ini_max - ini_min)
        df.loc[j, 'ini_nih'] = new_value

df.describe()
base_path = #path

def path_list(base_path, mode, dataframe):
    data_list = []
    sampler_list = np.array([])
    class_list = np.array([])
    for patient in sorted(os.listdir(os.path.join(base_path, mode))):
        name = patient.split('.')[0]
        for idx in range(df.shape[0]):
            
            if df.iloc[idx,0] == name:
                break
            else:
                continue
        if idx == df.shape[0]-1 and df.iloc[idx,0] != name :
            print(idx, df.shape[0]-1, df.iloc[idx,0], name)
            print(name)
            continue

        if df.iloc[idx, 0] == name:
            path = os.path.join(base_path, mode, patient)
            label = df.loc[idx, labels].to_numpy().astype(np.float32)
            if label[0] == 2:
                label = np.array([1]).astype(np.float32)


            case = {
            'image' : path,
            'feature' : df.loc[idx, features_reduction].to_numpy().astype(np.float32),
            'label' : label
            }
            if label.item() == 0:
                class_list = np.append(class_list, 0)
            else:
                class_list = np.append(class_list, 1)

            data_list.append(case)
    class_list = class_list.astype(np.uint8)
    labels_unique, counts = np.unique(class_list, return_counts = True)
    class_weights = [sum(counts) / c for c in counts]
    print('class_weights : 1/num_classes(0) & 1/num_classes(1)', class_weights)
    sampler_list = [class_weights[e] for e in class_list]
    return data_list, sampler_list


train_list, sampler_list_train = path_list(base_path, 'TRAIN/', df)
valid_list, _ = path_list(base_path, 'VALID/', df)
test_list, _ = path_list(base_path, 'TEST/', df)
print('train, valid, testset length :', len(train_list), len(valid_list), len(test_list))



class MyDataset(Dataset):
    def __init__(self, path_list, transform = None):
        self.path_list = path_list
        self.transform = transform
        
    def __getitem__(self, index):
        image = Image.open(self.path_list[index]['image'])
        image = image.convert("RGB")
        feature = torch.tensor(self.path_list[index]['feature']).float()
        label = torch.tensor(self.path_list[index]['label']).type(torch.uint8)

        if self.transform:
            image = self.transform(image)
        data = {'image' : image, 'feature' : feature, 'label' : label.item()}

        return data

    def __len__(self):
        return len(self.path_list)

tra = [
    transforms.RandomHorizontalFlip(), 
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.RandomAffine((-10,10), shear=10, scale=(0.9, 1.2)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

val = [
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

trainset = MyDataset(train_list, transform = transforms.Compose(tra))
validset = MyDataset(valid_list, transform = transforms.Compose(val))
   

sampler_train = WeightedRandomSampler(sampler_list_train, len(sampler_list_train), replacement = True)
train_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, sampler = sampler_train, num_workers = num_workers, pin_memory = True)
valid_loader = torch.utils.data.DataLoader(validset,batch_size=batch_size, shuffle = True, num_workers = num_workers, pin_memory = True)

one_batch = next(iter(train_loader))
fsize = one_batch['feature'].shape[1]

resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained = True)
new_resnet = nn.Sequential(*list(resnet.children())[:-1])

class Ensemble_weighted(nn.Module):
    def __init__(self, resnet):
        super(Ensemble_weighted, self).__init__()
        self.resnet = resnet

        self.mlp = nn.Sequential(
                    nn.Linear(fsize,fsize*8),
                    nn.ReLU(),
                    nn.Linear(fsize*8,fsize),
                    nn.ReLU()
                    )

        self.fc = nn.Sequential(
            nn.Linear(2048 + fsize, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self, image, feature):
        image = self.resnet(image)
        image = image.view(image.size(0), -1)
        feature = self.mlp(feature)
        concat = torch.cat((image,feature), dim = 1)
        output = self.fc(concat)
        return output
  


net = Ensemble_weighted(new_resnet)
net = net.to(device)

loss = torch.nn.CrossEntropyLoss() # loss
alg = torch.optim.Adam(net.parameters(),lr=learning_rate)

loss_train = np.array([])
loss_valid = np.array([])
accs_train = np.array([])
accs_valid = np.array([])

for epoch in range(num_epochs):

    net.train()
    i=0
    l_epoch = 0
    correct = 0
    l_epoch_val = 0
    for i_batch, item in enumerate(train_loader):
        i=i+1
        image, feature, y = item['image'].to(device), item['feature'].to(device), item['label'].type(torch.long).to(device)
        y_hat=net(image, feature)
        y_hat= F.softmax(y_hat, dim = 1)
        l=loss(y_hat,y)
        correct += (y_hat.argmax(dim=1)==y).sum()
        l_epoch+=l
        alg.zero_grad()
        l.backward()
        alg.step()
    loss_train = np.append(loss_train,l_epoch.cpu().detach().numpy()/i)
    accs_train = np.append(accs_train,correct.cpu()/np.float(len(trainset)))

    correct = 0
    i = 0
    net.eval()
    with torch.no_grad():
        for i_batch, item in enumerate(valid_loader):
            i +=1
            image, feature, y = item['image'].to(device), item['feature'].to(device), item['label'].to(device)
            y_hat=net(image, feature)
            y_hat= F.softmax(y_hat, dim = 1)
            l = loss(y_hat, y)
            correct += (y_hat.argmax(dim=1)==y).sum()
            l_epoch_val += l
    accs_valid = np.append(accs_valid,correct.cpu()/np.float(len(validset)))
    loss_valid = np.append(loss_valid, l_epoch_val.cpu().detach().numpy()/i)

    if True:
        fig = plt.figure(figsize = (12, 6))
        ax = fig.add_subplot(1,2,1)
        plt.plot(loss_train,label='train loss')
        plt.plot(loss_valid, label='valid loss')
        plt.legend(loc='lower left')
        plt.title('epoch: %d '%(epoch+1))

        ax = fig.add_subplot(1,2,2)
        plt.plot(accs_train,label='train accuracy')
        plt.plot(accs_valid,label='valid accuracy')
        plt.legend(loc='lower left')
        plt.pause(.0001)
        plt.show()

        print('train loss: ',loss_train[-1])
        print('valid loss: ', loss_valid[-1])
        print('train accuracy: ',accs_train[-1])
        print('valid accuracy: ',accs_valid[-1])



#torch.save(net.state_dict(), f'weight') parameter save
#
#net.load_state_dict(torch.load('weight'))


# 
# Inference
# 
base_path = '../../../../pred_png/' #png로 저장한 이미지 파일을 가져온다.
test_list, _ = path_list(base_path, 'TEST/', df)

batch_size = 32
testset = MyDataset(test_list, transform = transforms.Compose(val))
test_loader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False, num_workers = num_workers, pin_memory = True)


# In[37]:


correct = 0
label = np.array([])
pred = np.array([])
outs = np.array([])
for i_batch, item in enumerate(test_loader): ##train에서 test로 바꾸기
    image, feature, y = item['image'].to(device), item['feature'].to(device), item['label'].to(device)
    label = np.append(label,y.cpu())
    y_hat = net(image, feature)
    outs = np.append(outs,y_hat.softmax(dim=1).cpu().detach().numpy())
    pred = np.append(pred,y_hat.argmax(dim=1).cpu())
    correct += (y_hat.argmax(dim=1)==y).sum()
    
print('Accuracy : ', correct.cpu() / np.float(len(testset)))
print(outs.shape)
outs = outs.reshape(len(testset),2)


LABELS = ['Positive', 'Negative']
cfm=confusion_matrix(label,pred, labels = [1,0])
plt.figure(figsize = (9,7))
sns.set(font_scale=2)
sns.heatmap(np.int16(cfm), xticklabels = LABELS, yticklabels = LABELS, annot=True,cmap='Blues',fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.title('Confusion Matrix')
plt.show()


fpr, tpr, ths = roc_curve(label,outs[:,1])
aucval = auc(fpr, tpr)

plt.figure(figsize=(7,7))
plt.plot(fpr, tpr, 
         lw=3, label='ROC curve (area = %0.2f)' % aucval)
plt.plot([0, 1], [0, 1], color='red', lw=3,linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

