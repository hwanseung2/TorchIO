#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchio as tio
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import numpy as np
import enum
import multiprocessing
import random

from pathlib import Path
from PIL import Image
from unet import UNet
from tqdm import tqdm


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


seed = 42
random_seed(seed,True)

base_path = #학습시킬 데이터
mask_path = #학습에 매칭되는 마스크


def Dataset_load(base_path, mask_path,  mode):

    dataset_dir = Path(base_path)
    maskset_dir = Path(mask_path)

    images_dir = dataset_dir / mode / 'Img/'
    labels_dir = maskset_dir / mode / 'Mask/'

    image_paths = sorted(images_dir.glob('*.nii.gz'))
    label_paths = sorted(labels_dir.glob('*.nii.gz'))
    print(len(image_paths), len(label_paths))

    assert len(image_paths) == len(label_paths)

    MRI = 'mri'
    LABEL = 'label'
    subjects = []
    for (image_path, label_path) in zip(image_paths, label_paths):
        subject = tio.Subject(
            MRI = tio.ScalarImage(image_path),
            LABEL = tio.LabelMap(label_path),
        )

        subjects.append(subject)
    dataset = tio.SubjectsDataset(subjects)
    return dataset

trainset = Dataset_load(base_path, mask_path, 'train')
validset = Dataset_load(base_path, mask_path, 'valid')
testset = Dataset_load(base_path, mask_path, 'test')

from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation, 
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)

training_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(4),
    tio.CropOrPad((64,64,48)),
    tio.RandomMotion(p=0.2),
    tio.RandomBiasField(p=0.3),
    tio.RandomNoise(p=0.5),
    tio.RandomFlip(axes=(0,)),
    tio.RandomAffine(),
    ZNormalization(),
])

validation_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(4),
    tio.CropOrPad((64,64,48)),
    ZNormalization(),
])

training_set = tio.SubjectsDataset(subjects = trainset, transform = training_transform)
validation_set = tio.SubjectsDataset(subjects = validset, transform = validation_transform)

training_batch_size = 4
validation_batch_size = 2 * training_batch_size

training_loader = torch.utils.data.DataLoader(dataset = training_set, batch_size = training_batch_size, shuffle = True,
                                              num_workers=0)

validation_loader = torch.utils.data.DataLoader(dataset = validation_set, batch_size = validation_batch_size,
                                                num_workers=0)

train_batch = next(iter(training_loader))
valid_batch = next(iter(validation_loader))

print('training set : ', len(trainset), 'subjects')
print('validation set: ', len(validset), 'subjects\n')

for transform in training_transform:
    print(transform)

print('\n')

for transform in validation_transform:
    print(transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2,3,4

class Action(enum.Enum):
  TRAIN = 'Training'
  VALIDATE = 'Validation'

def prepare_batch(batch, device):
  inputs = batch['MRI'][tio.DATA].to(device)
  foreground = batch['LABEL'][tio.DATA].type(torch.float32).to(device)
  background = 1 - foreground
  targets = torch.cat((background, foreground), dim = CHANNELS_DIMENSION)
  return inputs, targets

def get_dice_score(output, target, epsilon = 1e-9):
  p0 = output
  g0 = target
  p1 = 1 - p0
  g1 = 1 - g0
  tp = (p0 * g0).sum(dim = SPATIAL_DIMENSIONS)
  fp = (p0 * g1).sum(dim = SPATIAL_DIMENSIONS)
  fn = (p1 * g0).sum(dim = SPATIAL_DIMENSIONS)
  num = 2 * tp
  denom = 2 * tp + fp + fn + epsilon
  dice_score = num / denom

  return dice_score

def get_dice_loss(output, target):
  return 1 - get_dice_score(output, target)

def forward(model, inputs):
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category = UserWarning)
    logits = model(inputs)
  return logits
def get_model_and_optimizer(device):
  model = UNet(
      in_channels = 1, 
      out_classes = 2,
      dimensions = 3,
      num_encoding_blocks = 4,
      out_channels_first_layer = 32,
      normalization = 'batch',
      padding = True,
      activation = 'PReLU',
  ).to(device)
  optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9)
  return model, optimizer

def run_epoch(epoch_idx, action, loader, model, optimizer):
  is_training = action == Action.TRAIN
  epoch_losses = []
  model.train(is_training)
  for batch_idx, batch in enumerate(tqdm(loader)):
    inputs, targets = prepare_batch(batch, device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(is_training):
      logits = forward(model, inputs)
      probabilities = F.softmax(logits, dim = CHANNELS_DIMENSION)
      batch_losses = get_dice_loss(probabilities, targets)
      batch_loss = batch_losses.mean()
      if is_training:
        batch_loss.backward()
        optimizer.step()
  
      
      epoch_losses.append(batch_loss.item())

  epoch_losses = np.array(epoch_losses)
  print(f'{action.value} mean loss : {epoch_losses.mean(): 0.3f}')
  return epoch_losses.mean()
  
def train(num_epochs, training_loader, validation_loader, model, optimizer, weights_stem):
  train_losses = np.array([])
  valid_losses = np.array([])
  valid_loss = run_epoch(0, Action.VALIDATE, validation_loader, model, optimizer)
  valid_losses = np.append(valid_losses, valid_loss)
  for epoch_idx in range(1, num_epochs+1):
    print('Starting epoch', epoch_idx)
    train_loss = run_epoch(epoch_idx, Action.TRAIN, training_loader, model, optimizer)
    valid_loss = run_epoch(epoch_idx, Action.VALIDATE, validation_loader, model, optimizer)
    train_losses = np.append(train_losses, train_loss)
    valid_losses = np.append(valid_losses, valid_loss)
    #torch.save(model.state_dict(), f'pth/{weights_stem}_epoch_{epoch_idx}.pth')

  return train_losses, valid_losses

train_whole_images = True
model, optimizer = get_model_and_optimizer(device)
num_epochs = 5
train_loss, valid_loss = train(num_epochs, training_loader, validation_loader, model, optimizer,'whole_images')

epoch_idx = 100
fig = plt.figure()
plt.plot(train_loss, label='train loss')
plt.plot(valid_loss, label='valid loss')
plt.legend(loc='lower left')
plt.title('epoch: %d '%(epoch_idx))
plt.pause(.0001)

#torch.save(model.state_dict(), f'1st_e100_lr2e-3_SGD.pth') 
model.load_state_dict(torch.load('1st_e100_lr2e-3_SGD.pth'))

'''
Inference
'''
test_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(4),
    tio.CropOrPad((64,64,48)),
    ZNormalization(),
])

test_set = tio.SubjectsDataset(subjects = testset, transform = test_transform)
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = validation_batch_size, shuffle = False, num_workers = 0)

test_loss = []
for batch_idx, batch in enumerate(tqdm(test_loader)):
    inputs, targets = prepare_batch(batch, device)
    with torch.no_grad():
        outputs = forward(model, inputs)
        probabilities = F.softmax(outputs, dim = CHANNELS_DIMENSION)
        batch_losses = get_dice_loss(probabilities, targets)
        test_loss.append(batch_losses.mean().item())

print('\n', test_loss)

print('testset diceloss : ', sum(test_loss) / len(test_loss))

'''
inference
'''

# 
# 니프티 파일 만들기 train 5401개 / valid 660 개 / test 681개
# 
import nibabel as nib
base_path = '/home/hsyoo/promedius/brainmr_cnuh/dataset/nifti/test/Img/'
output_path = '../../../../old_pred_label/test/'
for item in sorted(os.listdir(base_path)):
    print(item)
    path = os.path.join(base_path, item)
    subject = tio.Subject(MRI=tio.ScalarImage(path))
    transformed = test_transform(subject)

    inputs = transformed['MRI']['data'].unsqueeze(0).to(device)
    outputs = forward(model, inputs)
    probabilities = F.softmax(outputs, dim = CHANNELS_DIMENSION)
    labels = probabilities.argmax(dim=CHANNELS_DIMENSION)
    tf_z = transformed['MRI']['data'].shape[-1]
    sub_z = subject['MRI']['data'].shape[-1]
    z_start = ((tf_z - sub_z) // 2)
    
    pred_label = labels[0,:,:,z_start : z_start + sub_z].cpu().detach().numpy()
    nifti = nib.Nifti1Image(pred_label, subject['MRI']['affine'])

    output_name = os.path.join(output_path + item)
    print(output_path)
    #nib.save(nifti, output_name)

    print(pred_label.shape, '\n\n\n')
