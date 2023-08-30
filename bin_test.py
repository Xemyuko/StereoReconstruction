# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:09:28 2023

@author: myuey
"""
import scripts as scr
import numpy as np
from tqdm import tqdm
import numba
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

scr.display_stereo(maskL[0],maskR[0])

maskL = scr.boost_list(maskL, scale_factor,xOffset1, xOffset2, yOffset1, yOffset2)
maskR = scr.boost_list(maskR, scale_factor,xOffset1, xOffset2, yOffset1, yOffset2)

#display the offsets on the masked images
scr.display_stereo(maskL[0],maskR[0])
maskL = np.asarray(maskL, dtype = 'uint8')
maskR = np.asarray(maskR, dtype = 'uint8')
#binary conversion
bin_thresh = 70
maskL = scr.bin_convert_arr(maskL, 150)
maskR = scr.bin_convert_arr(maskR, 150)
scr.display_stereo(maskL[0],maskR[0])
@numba.jit(nopython=True)
def cor_acc_pix(Gi,x,y,n, xLim, maskR, xOffset1, xOffset2):
    min_cor = 10000
    max_index = -1
    max_mod = [0,0] #default to no change
    #Search the entire line    
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = maskR[:,y,xi]
        cor = np.sum((Gi-Gt)**2)            
        if cor < min_cor:
            min_cor = cor
            max_index = xi
    #search surroundings of found best match
    Gup = maskR[:,y-1, max_index]    
    cor = np.sum((Gi-Gup)**2)        
    if cor < min_cor:
        min_cor = cor
        max_mod = [-1,0]
    
    Gdn = maskR[:,y+1, max_index]
    cor = np.sum((Gi-Gdn)**2)     
    if cor < min_cor:
        min_cor = cor
        max_mod = [1,0]        
    return max_index,min_cor,max_mod

@numba.jit(nopython=True)
def cor_sur_pix(Gi,Gi_card,Gi_diag,x,y,n, xLim, maskR, xOffset1, xOffset2):
    min_cor = 10000
    max_index = -1
    max_mod = [0,0] #default to no change
    #[N,S,W,E]
    #coord_card = [(-1,0),(1,0),(0,-1),(0,1)]
    #[NW,SE,NE,SW]
    #coord_diag = [(-1,-1),(1,1),(-1,1),(1,-1)]
    car_num = 4
    
    for xi in range(xOffset1, xLim-xOffset2):
        #define points
        G_cent =  maskR[:,y,xi]
        '''
        G_card = [maskR[:,y-1,xi],maskR[:,y+1,xi],maskR[:,y,xi-1],
                      maskR[:,y,xi+1]]
        '''
        G_diag = [maskR[:,y-1,xi-1],maskR[:,y+1,xi+1],maskR[:,y-1,xi+1],
                      maskR[:,y+1,xi-1]]
        cor = np.sum((Gi-G_cent)**2) 
        '''
        for a in range(car_num):
            cor += np.sum((Gi_card[a]-G_card[a])**2)
         '''   
        for a in range(car_num):
            cor += np.sum((Gi_diag[a]-G_diag[a])**2)
        if cor < min_cor:
            min_cor = cor
            max_index = xi
    
    Gup = maskR[:,y-1, max_index]    
    cor = np.sum((Gi-Gup)**2)
    '''
    G_card = [maskR[:,y-2,xi],maskR[:,y,xi],maskR[:,y-1,xi-1],
                  maskR[:,y-1,xi+1]]
    '''
    G_diag = [maskR[:,y-2,xi-1],maskR[:,y,xi+1],maskR[:,y-2,xi+1],
                  maskR[:,y,xi-1]]
    '''
    for a in range(car_num):
        cor += np.sum((Gi_card[a]-G_card[a])**2)
        '''
    for a in range(car_num):
        cor += np.sum((Gi_diag[a]-G_diag[a])**2)
    if cor < min_cor:
        min_cor = cor
        max_mod = [-1,0]    
    Gdn = maskR[:,y+1, max_index]    
    cor = np.sum((Gi-Gdn)**2)
    '''
    G_card = [maskR[:,y,xi],maskR[:,y+2,xi],maskR[:,y+1,xi-1],
                  maskR[:,y+1,xi+1]]
    '''
    G_diag = [maskR[:,y,xi-1],maskR[:,y+2,xi+1],maskR[:,y,xi+1],
                  maskR[:,y+2,xi-1]]
    '''
    for a in range(car_num):
        cor += np.sum((Gi_card[a]-G_card[a])**2)
        '''
    for a in range(car_num):
        cor += np.sum((Gi_diag[a]-G_diag[a])**2)
    if cor < min_cor:
        min_cor = cor
        max_mod = [-1,0]
        
    return max_index,min_cor,max_mod
#duplicate comparison and correlation thresholding, run when trying to add points to results
def compare_cor(res_list, entry_val, threshold, recon = True):
    remove_flag = False
    pos_remove = 0
    entry_flag = False
    counter = 0
    if(recon):
        if(entry_val[1] < 0 or entry_val[2] < threshold):
            return pos_remove,remove_flag,entry_flag
    else:
        if(entry_val[1] < 0):
            return pos_remove,remove_flag,entry_flag
    for i in range(len(res_list)):       
        
        if(res_list[i][1] == entry_val[1] and res_list[i][3][0] - entry_val[3][0] < float_epsilon and
           res_list[i][3][1] - entry_val[3][1] < float_epsilon):
            #duplicate found, check correlation values and mark index for removal
            remove_flag = (res_list[i][2] > entry_val[2])
            pos_remove = i
            break
        else:
            counter+=1
    #end of list reached, no duplicates found, entry is valid
    if(counter == len(res_list)):
        entry_flag = True
    return pos_remove,remove_flag,entry_flag

    
interval = 1           
rect_res = []
n = len(imgL)

for y in tqdm(range(yOffset1, yLim-yOffset2)):
    res_y = []
    for x in range(xOffset1, xLim-xOffset2, interval):
        Gi = maskL[:,y,x]
        if(np.sum(Gi) != 0): #dont match fully dark slices
            Gi_card = np.asarray([maskL[:,y-1,x],maskL[:,y+1,x],maskL[:,y,x-1],
                          maskL[:,y,x+1]])
            Gi_diag = np.asarray([maskL[:,y-1,x-1],maskL[:,y+1,x+1],maskL[:,y-1,x+1],
                          maskL[:,y+1,x-1]])
            x_match,cor_val,subpix = cor_sur_pix(Gi,Gi_card,Gi_diag,
                                                 x,y,n, xLim, maskR, xOffset1, xOffset2)
                
            pos_remove, remove_flag, entry_flag = compare_cor(res_y,
                                                              [x,x_match, cor_val, subpix, y], thresh)
            if(remove_flag):
                res_y.pop(pos_remove)
                res_y.append([x,x_match, cor_val, subpix, y])
            elif(entry_flag):
                res_y.append([x,x_match, cor_val, subpix, y])
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
        sL = HL[2,0]*q[0] + HL[2,1] * (q[4]+yOffset1) + HL[2,2]
        pL = hL_inv @ np.asarray([[q[0]],[q[4]+yOffset1],[sL]])
        sR = HR[2,0]*(q[1] + q[3][1]) + HR[2,1] * (q[4]+yOffset1+q[3][0]) + HR[2,2]
        pR = hR_inv @ np.asarray([[q[1]+ q[3][1]],[q[4]+yOffset1+q[3][0]],[sR]])
        ptsL.append([pL[0,0],pL[1,0],pL[2,0]])
        ptsR.append([pR[0,0],pR[1,0],pR[2,0]])

ptsL = []
ptsR = []
for a in range(len(rect_res)):
    b=rect_res[a]
    for q in b:
        pLy = a + yOffset1
        pLx = q[0]
        ptsL.append([pLx, pLy, 0])
        
        pRx = q[1]
        pRy = a + yOffset1 + q[3][0]
        ptsR.append([pRx, pRy, 0])
#Triangulate 3D positions from point lists
#take 2D
ptsL = scr.conv_pts(ptsL)
ptsR = scr.conv_pts(ptsR)
col_arr = scr.gen_color_arr(imgL[0],imgR[0], ptsL, ptsR)
tri_res = scr.triangulate_list(ptsL,ptsR, r_vec, t_vec, kL_inv, kR_inv)
#Convert numpy arrays to ply point cloud file
scr.convert_np_ply(np.asarray(tri_res), col_arr,"test-bin.ply", overwrite=True)
res_map = np.zeros((maskL.shape[1],maskL.shape[2]), dtype = 'uint8')
for i in range(len(rect_res)):
    b = rect_res[i]
    for j in b:
        res_map[i+yOffset1,j[0]] = j[2]*255
cv2.imwrite("binmaptest.png", res_map)