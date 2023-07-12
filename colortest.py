# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:37:41 2023

@author: Admin
"""

import numpy as np
import scripts as scr
from tqdm import tqdm
import numba
import matplotlib.pyplot as plt
import ncc_core

#Load camera matrices and images
folder_statue = "./test_data/mouse/"
matrix_folder = "matrix_folder/"
left_folder = "camera_L/"
right_folder = "camera_R/"

tmod = 1
kL,kR,r_vec,t_vec = scr.initial_load(tmod, folder_statue + matrix_folder)
kL_inv = np.linalg.inv(kL)
kR_inv = np.linalg.inv(kR)
imgL = None
imgR = None

colortesting = False
fmat_load = False

if colortesting:
    imgL,imgR = scr.load_color_split(folderL = folder_statue+left_folder, folderR = folder_statue+right_folder)
else: 
    imgL,imgR = scr.load_images(folderL = folder_statue+left_folder, folderR = folder_statue+right_folder)
    
imshape = imgL[0].shape
print(imshape)
F = None
if fmat_load:
    F = np.loadtxt(matrix_folder + "f.txt", skiprows=2, delimiter = " ")
else: 
    pts1b,pts2b,colb, F = scr.feature_corr(imgL[0],imgR[0])

rectL,rectR = scr.rectify_lists(imgL,imgR, F)
avgL = np.asarray(rectL).mean(axis=(0))
avgR = np.asarray(rectR).mean(axis=(0))

#Background filter
thresh_val = 10

maskL = scr.mask_inten_list(avgL, rectL, thresh_val)
maskR = scr.mask_inten_list(avgR, rectR, thresh_val)
maskL = np.asarray(maskL)
maskR = np.asarray(maskR)

#define constants for window
xLim = imshape[1]
yLim = imshape[0]
xOffset = 1
yOffset = 1

#define constants for correlation
default_thresh = 0.9
float_epsilon = 1e-9
default_interp = 3

