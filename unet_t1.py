# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:03:46 2025

@author: myuey
"""

import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt
import scripts as scr
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision

x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())

def sp_noise(image,prob, mode = False):
    '''
    
    prob: Probability of the noise
    mode: salt&pepper or only pepper - simulates darkening from low exposure time
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                if mode:
                    output[i][j] = 1 
                else:
                    output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output




def test_noi_gen():
    #load image
    #load image pair
    folder1 = './test_data/250221_Cudatest/pos9/'
    #folder1 = './test_data/testset1/bulb-multi/b1/'
    imgLc,imgRc = scr.load_images_1_dir(folder1, 'cam1', 'cam2', ext = '.jpg', colorIm = True)
    imgL,imgR = scr.load_images_1_dir(folder1, 'cam1', 'cam2', ext = '.jpg', colorIm = False)

    n1 = sp_noise(imgLc[0],0.3, True)
    scr.display_stereo(imgLc[0], n1)
    
    res = []
    for i in tqdm(imgLc):
        res.append(sp_noise(i, 0.2, True))
        res.append(sp_noise(i, 0.3, True))

test_noi_gen()