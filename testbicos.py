# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:55:18 2025

@author: myuey
"""

import numpy as np
import scripts as scr
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numba
import itertools as itt
import scipy.signal as sig
#used for comparing floating point numbers to avoid numerical errors
float_epsilon = 1e-9


def biconv3(imgs, n = 8, comN = 4):
    #Add neighbors together, then compare combinations
    imshape = imgs[0].shape
    imgs1a = np.zeros((n,imshape[0],imshape[1]))
    for a in range(n):
        imgs1a[a,:,:]  = imgs[a,:,:]

    imgs1b = np.zeros((n,imshape[0],imshape[1]))
    for i in range(1,n):
        imgs1b[i-1,:,:] = imgs[i-1,:,:] + imgs[i,:,:]
        
    combs = list(itt.combinations(range(1, n + 1), comN))
    perm_combs = []

    for comb in combs:
        perm_combs.extend(itt.permutations(comb))

    perm_combs = np.array(sorted(perm_combs))
    # Remove unwanted permutations
    perm_combs = perm_combs[(perm_combs[:, 2] <= perm_combs[:, 3]) &
                         (perm_combs[:, 0] <= perm_combs[:, 1]) &
                         (perm_combs[:, 0] <= perm_combs[:, 2])]         
     
    bilength = perm_combs.shape[0]
    res_stack1 = np.zeros((bilength,imshape[0],imshape[1]),dtype = 'int8')
    for indval in tqdm(range(bilength)):
        i, j, k, l = perm_combs[indval]
        res_stack1[indval,:,:] = (imgs1b[i-1,:,:] + imgs1b[j-1,:,:]) > (imgs1b[k-1,:,:] + imgs1b[l-1,:,:])

    return res_stack1
    

def biconv1(imgs, n = 8):
    #Compare with average
    imshape = imgs[0].shape
    imgs1a = np.zeros((n,imshape[0],imshape[1]))
    imgs1b = np.zeros((n,imshape[0],imshape[1]))
    for a in range(n):
        imgs1a[a,:,:]  = imgs[a,:,:]
        
    avg_img = imgs1a.mean(axis=(0))
    for b in range(n):
        imgs1b[b,:,:] = imgs1a[b,:,:] > avg_img
    return imgs1b
def biconv2(imgs, n = 8, comN = 4):
    #Compare combinations of images, each image with every other image
    imshape = imgs[0].shape
    imgs1a = np.zeros((n,imshape[0],imshape[1]))

    for a in range(n):
        imgs1a[a,:,:]  = imgs[a,:,:]

    combs = list(itt.combinations(range(1, n + 1), comN))
    perm_combs = []

    for comb in combs:
        perm_combs.extend(itt.permutations(comb))

    perm_combs = np.array(sorted(perm_combs))
   # Remove unwanted permutations
    perm_combs = perm_combs[(perm_combs[:, 2] <= perm_combs[:, 3]) &
                        (perm_combs[:, 0] <= perm_combs[:, 1]) &
                        (perm_combs[:, 0] <= perm_combs[:, 2])]         
    
    bilength = perm_combs.shape[0]
    res_stack1 = np.zeros((bilength,imshape[0],imshape[1]),dtype = 'int8')
    for indval in tqdm(range(bilength)):
        i, j, k, l = perm_combs[indval]
        res_stack1[indval,:,:] = (imgs1a[i-1,:,:] + imgs1a[j-1,:,:]) > (imgs1a[k-1,:,:] + imgs1a[l-1,:,:])

    return res_stack1
def biconv4(imgs, n = 8, comN = 4):
    #Compare to average and also compare combinations of images
    imshape = imgs[0].shape
    imgs1a = np.zeros((n,imshape[0],imshape[1]))
    imgs1b = np.zeros((n,imshape[0],imshape[1]))
    for a in range(n):
        imgs1a[a,:,:]  = imgs[a,:,:]
        
    avg_img = imgs1a.mean(axis=(0))
    for b in range(n):
        imgs1b[b,:,:] = imgs1a[b,:,:] > avg_img
        
@numba.jit(nopython=True)
def bi_pix(Gi,y,n, xLim, maskR, xOffset1, xOffset2):
    max_cor = 0.0
    max_index = -1
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = maskR[:,y,xi]
        chkres = Gi-Gt
        chk = np.sum(chkres == 0)/n
        if(chk > max_cor):
            max_index = xi
            max_cor = chk
    return max_index,max_cor

def comcor1(res_list, entry_val, threshold):
    remove_flag = False
    pos_remove = 0
    entry_flag = False
    counter = 0
    if(entry_val[1] < 0 or entry_val[2] < threshold):
        return pos_remove,remove_flag,entry_flag
    for i in range(len(res_list)):       
        
        if(res_list[i][1] == entry_val[1]):
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


def run_bicos():
    #load images
    imgFolder = './test_data/testset1/bulb-multi/b1/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgs1,imgs2 = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    col_refL, col_refR = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd,colorIm = True)
    #apply filter
    thresh1 = 30
    imgs1 = np.asarray(scr.mask_inten_list(imgs1,thresh1))
    imgs2 = np.asarray(scr.mask_inten_list(imgs2,thresh1))
    #load matrices
    mat_folder = './test_data/testset1/matrices/'
    kL, kR, r, t = scr.load_mats(mat_folder) 
    f = np.loadtxt(mat_folder + 'f.txt', delimiter = ' ', skiprows = 2)
    #rectify images
    v,w, H1, H2 = scr.rectify_pair(imgs1[0], imgs2[0], f)
    imgs1,imgs2 = scr.rectify_lists(imgs1,imgs2,f)
    imgs1 = np.asarray(imgs1)
    imgs2 = np.asarray(imgs2)
    imshape = imgs1[0].shape
    n = 8
    #binary conversion
    imgs1 = biconv2(imgs1, n = n)
    imgs2 = biconv2(imgs2, n = n)
    #Take left and compare to right side to find matches
    offset = 10
    rect_res = []
    xLim = imshape[1]
    yLim = imshape[0]
    for y in tqdm(range(offset, yLim-offset)):
        res_y = []
        for x in range(offset, xLim-offset):
            Gi = imgs1[:,y,x].astype('uint8')
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                x_match,cor_val= bi_pix(Gi,y,n, xLim, imgs2, offset, offset)

                pos_remove, remove_flag, entry_flag = comcor1(res_y,
                                                                  [x,x_match, cor_val, y], 0.1)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, y])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, y])
        
        rect_res.append(res_y)
    #Convert matched points from rectified space back to normal space
    
    hL_inv = np.linalg.inv(H1)
    hR_inv = np.linalg.inv(H2)
    ptsL = []
    ptsR = []
    for a in range(len(rect_res)):
        b = rect_res[a]
        
        for q in b:
            print(len(q))
            xL = q[0]
            y = q[4]
            xR = q[1]
            xL_u = (hL_inv[0,0]*xL + hL_inv[0,1] * y + hL_inv[0,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * y + hL_inv[2,2])
            yL_u = (hL_inv[1,0]*xL + hL_inv[1,1] * y + hL_inv[1,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * y + hL_inv[2,2])
            xR_u = (hR_inv[0,0]*xR + hR_inv[0,1] * y + hR_inv[0,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * y + hR_inv[2,2])
            yR_u = (hR_inv[1,0]*xR + hR_inv[1,1] * y + hR_inv[1,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * y + hR_inv[2,2])
            ptsL.append([xL_u,yL_u])
            ptsR.append([xR_u,yR_u])
            
    col_ptsL = np.around(ptsL,0).astype('uint16')
    col_ptsR = np.around(ptsR,0).astype('uint16')        
    col_arr = scr.get_color(col_refL, col_refR, col_ptsL, col_ptsR)
    tri_res = scr.triangulate_list(ptsL,ptsR, r, t, kL, kR)
    
    scr.convert_np_ply(np.asarray(tri_res), col_arr,"tbicos.ply")
    
run_bicos()
