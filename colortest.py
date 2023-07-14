# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:37:41 2023

@author: Admin
"""

import numpy as np
import scripts as scr
from tqdm import tqdm
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
fmat_coeff = 0.6

if colortesting:
    imgL,imgR = scr.load_color_split(folderL = folder_statue+left_folder, folderR = folder_statue+right_folder)
else: 
    imgL,imgR = scr.load_images(folderL = folder_statue+left_folder, folderR = folder_statue+right_folder)
    
imshape = imgL[0].shape
print("Image Shape: " +str(imshape))

F = None
if fmat_load:
    F = np.loadtxt(matrix_folder + "f.txt", skiprows=2, delimiter = " ")
else: 
    F = scr.find_f_mat(imgL,imgR, thresh = fmat_coeff, precise = True)

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
thresh = 0.9
float_epsilon = 1e-9
interp = 3

rect_res = []
n = len(imgL)
for y in tqdm(range(yOffset, yLim-yOffset)):
    res_y = []
    for x in range(xOffset, xLim-xOffset):
        Gi = maskL[:,y,x]
        if(np.sum(Gi) != 0): #dont match fully dark slices
            x_match,cor_val,subpix = ncc_core.cor_acc_linear(Gi,x,y,n, xLim, maskR, xOffset, interp)
                
            pos_remove, remove_flag, entry_flag = ncc_core.compare_cor(res_y,
                                                              [x,x_match, cor_val, subpix], thresh)
            if(remove_flag):
                res_y.pop(pos_remove)
                res_y.append([x,x_match, cor_val, subpix])
            elif(entry_flag):
                res_y.append([x,x_match, cor_val, subpix])
    rect_res.append(res_y)
    

#Convert matched points from rectified space back to normal space
im_a,im_b,HL,HR = scr.rectify_pair(imgL[0],imgR[0], F)
hL_inv = np.linalg.inv(HL)
hR_inv = np.linalg.inv(HR)
ptsL = []
ptsR = []
for a in range(len(rect_res)):
    b = rect_res[a]
    for q in b:
        sL = HL[2,0]*q[0] + HL[2,1] * (a+yOffset) + HL[2,2]
        pL = hL_inv @ np.asarray([[q[0]],[a+yOffset],[sL]])
        sR = HR[2,0]*(q[1] + q[3][1]) + HR[2,1] * (a+yOffset+q[3][0]) + HR[2,2]
        pR = hR_inv @ np.asarray([[q[1]+ q[3][1]],[a+yOffset+q[3][0]],[sR]])
        ptsL.append([pL[0,0],pL[1,0],pL[2,0]])
        ptsR.append([pR[0,0],pR[1,0],pR[2,0]])


#Triangulate 3D positions from point lists
#take 2D
ptsL = scr.conv_pts(ptsL)
ptsR = scr.conv_pts(ptsR)
col_arr = scr.gen_color_arr_black(len(ptsL))
tri_res = scr.triangulate_list(ptsL,ptsR, r_vec, t_vec, kL_inv, kR_inv)
print()
#Convert numpy arrays to ply point cloud file
scr.convert_np_ply(np.asarray(tri_res), col_arr,"colortest")