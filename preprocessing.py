#!/usr/bin/env python

import nibabel as nib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os

base_path =  # 원본 니프티파일이 존재하는 곳 / image파일을 가져올 예정
pred_path = #원본 니프티로 부터 png파일을 만든 path
output_path = #아웃풋이 나오는 path
mode = # mode는 train, valid, test로 나누어서 설정하였다.

for patient in sorted(os.listdir(os.path.join(pred_path))):
    if patient.split('.')[0].split('_')[-1] == 'init': #init만 가져오도록 여기서 설정이 다 돼 있다!
        pred = nib.load(os.path.join(pred_path, patient)).get_fdata()
        img = nib.load(os.path.join(base_path, mode, 'Img/', patient)).get_fdata()
        if pred.shape[0] == 64 and pred.shape[1] == 64:
            max = -1
            idx = 0
            for z in range(pred.shape[-1]):
                temp = pred[:,:,z]
                sum = temp[temp>=0.5].sum()
                count = len(temp[temp>=0.5])+0.0000001
                avg = sum/count
                if avg > max:
                    max = avg
                    idx = z

            print(patient, idx)
            if idx ==0:
                continue
            data = img[:,:, idx-1 : idx+2]
            data = data / data.max()
            data = data * 255
            data = Image.fromarray(data.astype(np.uint8))
            name = patient.split('_')[2]
        


    else:
        continue
