# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:09:28 2023

@author: myuey
"""
import scripts as scr
import numpy as np
from tqdm import tqdm
import numba
import matplotlib.pyplot as plt
import cv2
#target loading folders
folder_statue = "./test_data/statue/"
matrix_folder = "matrix_folder/"
left_folder = "camera_L/"
right_folder = "camera_R/"
#offset values
xOffset1 = 1
xOffset2 = 1
yOffset1 = 1
yOffset2 = 1

tmod = 0.416657633
#tmod = 1

#define constants for correlation
thresh = 0.9
float_epsilon = 1e-9
interp = 3
#initial load and setup
kL,kR,r_vec,t_vec = scr.initial_load(tmod, folder_statue + matrix_folder)
kL_inv = np.linalg.inv(kL)
kR_inv = np.linalg.inv(kR)
#Load images
imgL,imgR = scr.load_images(folderL = folder_statue+left_folder, folderR = folder_statue+right_folder)
imshape = imgL[0].shape
xLim = imshape[1]
yLim = imshape[0]
#rectify images
F = scr.find_f_mat(imgL[0],imgR[0])
rectL,rectR = scr.rectify_lists(imgL,imgR, F)
avgL = np.asarray(rectL).mean(axis=(0))
avgR = np.asarray(rectR).mean(axis=(0))
#Background filter
thresh_val = 30
maskL = scr.mask_avg_list(avgL, rectL, thresh_val)
maskR = scr.mask_avg_list(avgR, rectR, thresh_val)
#boost signal for binary

scale_factor = 2
maskL = np.asarray(maskL, dtype = 'uint8')
maskR = np.asarray(maskR, dtype = 'uint8')
'''
fig = scr.create_stereo_offset_fig(maskL[0],maskR[0],xOffset1,xOffset2,yOffset1,yOffset2)
plt.show()

maskL = scr.boost_list(maskL, scale_factor,xOffset1, xOffset2, yOffset1, yOffset2)
maskR = scr.boost_list(maskR, scale_factor,xOffset1, xOffset2, yOffset1, yOffset2)

print(np.average(maskL[0]))
print(np.max(maskL[0]))
#display the offsets on the masked images
fig = scr.create_stereo_offset_fig(maskL[0],maskR[0],xOffset1,xOffset2,yOffset1,yOffset2)
plt.show()
'''
#binary conversion
maskL = scr.bin_convert_arr(maskL, 20)
maskR = scr.bin_convert_arr(maskR, 20)

fig = scr.create_stereo_offset_fig(maskL[0],maskR[0],xOffset1,xOffset2,yOffset1,yOffset2)
plt.show()