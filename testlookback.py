# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:41:06 2025

@author: Admin
"""

import numpy as np
import scripts as scr
import matplotlib.pyplot as plt

from tqdm import tqdm
import cv2
import json
import numba
import time 
import threading as thr
from numba import cuda as cu
import time
import os
import ncc_core as ncc
import confighandler as chand
import scipy.signal as sig
from scipy.interpolate import Rbf
import scipy.interpolate
import scipy.linalg as sclin
import tkinter as tk
import inspect
import csv
import itertools as itt
#used for comparing floating point numbers to avoid numerical errors
float_epsilon = 1e-9

def compare_cor(res_list, entry_val, threshold, recon = True):
    '''
    Checks proposed additions to the list of correlated points for duplicates, threshold requirements, and existing matches
    
    
    Parameters
    ----------
    res_list : list of entries
        Existing list of entries
    entry_val : list of values in the format: [x,x_match, cor_val, subpix, y]
        x : Left image x value of pixel stack
        x_match: Matched right image pixel stack x value
        cor_val: Correlation score for match
        subpix: Subpixel interpolation coordinates
        y: y value of the rectified line that x and x-match are found in
    threshold : float
        Minimum correlation value needed to be added to list of results
    recon : boolean, optional
        Controls if the threshold needs to be met, which is not needed for making a correlation map. The default is True.

    Returns
    -------
    pos_remove : integer
        position of entry to remove from list
    remove_flag : boolean
        True if an entry needs to be removed
    entry_flag : boolean
        True if entry_val is a valid addition to the result list

    '''
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

    
@numba.jit(nopython=True)
def ncc_pix(Gi,y,n, xLim, maskR, xOffset1, xOffset2):
    '''
    NCC point correlation function with no subpixel interpolation

    Parameters
    ----------
    Gi : Numpy array
        Vector with grayscale values of pixel stack to match with
    y : integer
        y position of row of interest
    n : integer
        number of images in image stack
    xLim : integer
        Maximum number for x-dimension of images
    maskR : 2D image stack
        vertical stack of 2D numpy array image data
    xOffset1 : integer
        Offset from left side of image stack to start looking from
    xOffset2 : integer
        Offset from right side of image stack to stop looking at

    Returns
    -------
    max_index : integer
        identified best matching x coordinate
    max_cor : float
        correlation value of best matching coordinate
    max_mod : list of floats wth 2 entries
        modifier to apply to best matching coordinate if the actual best is above or below.

    '''

    max_cor = 0.0
    max_index = -1
    max_mod = [0,0]
    agi = np.sum(Gi)/n
    
    val_i = np.sum((Gi-agi)**2)
    
    #Search the entire line    
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = maskR[:,y,xi]
        agt = np.sum(Gt)/n        
        val_t = np.sum((Gt-agt)**2)
        if(val_i > float_epsilon and val_t > float_epsilon): 
            cor = np.sum((Gi-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))              
            if cor > max_cor:
                max_cor = cor
                max_index = xi
    #search surroundings of found best match
    Gup = maskR[:,y-1, max_index]
    agup = np.sum(Gup)/n
    val_up = np.sum((Gup-agup)**2)
    if(val_i > float_epsilon and val_up > float_epsilon): 
        cor = np.sum((Gi-agi)*(Gup - agup))/(np.sqrt(val_i*val_up))              
        if cor > max_cor:
           max_cor = cor
           max_mod = [-1,0]
    
    Gdn = maskR[:,y+1, max_index]
    agdn = np.sum(Gdn)/n
    val_dn = np.sum((Gdn-agdn)**2)
    if(val_i > float_epsilon and val_dn > float_epsilon): 
        cor = np.sum((Gi-agi)*(Gdn - agdn))/(np.sqrt(val_i*val_dn))              
        if cor > max_cor:
            max_cor = cor
            max_mod = [1,0]  
    return max_index,max_cor,max_mod

@numba.jit(nopython=True)
def bcc_pix(Gi,y,n, xLim, maskR, xOffset1, xOffset2):
    max_cor = 0.0
    max_index = -1
    max_mod = [0,0] #default to no change
    
    
    #Search the entire line    
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = maskR[:,y,xi]
        vc = np.sum(np.absolute(Gi-Gt))/n
        cor = 1-vc

     

        if cor > max_cor:
            max_cor = cor
            max_index = xi
    #search surroundings of found best match
    Gup = maskR[:,y-1, max_index]       
    vc = np.sum(np.absolute(Gi-Gup))/n

    cor = 1-vc

 

    if cor > max_cor:
        max_cor = cor
        max_mod = [-1,0]
        
    Gdn = maskR[:,y+1, max_index]       
    vc = np.sum(np.absolute(Gi-Gdn))/n

    cor = 1-vc

 

    if cor > max_cor:
        max_cor = cor
        max_mod = [1,0]

    return max_index,max_cor,max_mod
def biconv1(imgs, n = 8, comN = 4):
    
    imshape = imgs[0].shape
    imgs1a = np.zeros((n,imshape[0],imshape[1]))

    for a in range(n):
        imgs1a[a]  = imgs[a]

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
    res_stack1 = np.zeros((bilength,imshape[0],imshape[1]),dtype = 'uint8')
    for indval in tqdm(range(bilength)):
        i, j, k, l = perm_combs[indval]
        res_stack1[indval] = (imgs1a[i-1] + imgs1a[j-1]) > (imgs1a[k-1] + imgs1a[l-1])

    return res_stack1

def unpack_rect_res(listin):
    pts1 = []
    pts2 = []
    cor = []
    
    #[x,x_match, cor_val, subpix, y]
    for i in listin:
        for j in i:
            y1 = j[4] + j[3][0]
            x1 = j[0]+j[3][1]
            y2 = j[4]+ j[3][0]
            x2 = j[1]+j[3][1]
            pts1.append([y1,x1])
            pts2.append([y2,x2])
            cor.append(j[2])
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    cor = np.array(cor)
    
    
    return pts1,pts2,cor

def test_bcc_lookback():
    #load images
    imgFolder = './test_data/testset1/bulb/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgs1,imgs2 = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    col_refL, col_refR = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd,colorIm = True)
    n= imgs1.shape[0]
    #apply filter
    thresh1 = 10
    imgs1 = np.asarray(scr.mask_inten_list(imgs1,thresh1))
    imgs2 = np.asarray(scr.mask_inten_list(imgs2,thresh1))
    imshape = imgs1[0].shape
    
    
    print(imshape)
    #pull a small number of images for testing
    n =12
    comN = 4
    imgs1a = np.zeros((n,imshape[0],imshape[1]))
    imgs2a = np.zeros((n,imshape[0],imshape[1]))
    for a in range(n):
        imgs1a[a,:,:]  = imgs1[a,:,:]
        imgs2a[a,:,:] = imgs2[a,:,:]
        
        
    #load matrices
    mat_folder = './test_data/testset1/matrices/'
    kL, kR, r, t = scr.load_mats(mat_folder) 
    f = np.loadtxt(mat_folder + 'f.txt', delimiter = ' ', skiprows = 2)
    #rectify images
    v,w, H1, H2 = scr.rectify_pair(imgs1[0], imgs2[0], f)
    imgs1a,imgs2a = scr.rectify_lists(imgs1,imgs2,f)
    imgs1a = np.asarray(imgs1a)
    imgs2a = np.asarray(imgs2a)
    plt.imshow(imgs1a[0])
    plt.show()
    res_stack1 = biconv1(imgs1a,n,comN)
    res_stack2 = biconv1(imgs2a,n,comN)
    plt.imshow(res_stack1[0])
    plt.show()
    print(res_stack1.shape)
    #run correlation search on stacks
    offset = 10
    rect_res = []
    xLim = imshape[1]
    yLim = imshape[0]
    threshc = 0.0
    #Take left and compare to right side to find matches
    for y in tqdm(range(offset, yLim-offset)):
        res_y = []
        for x in range(offset, xLim-offset):
            Gi = res_stack1[:,y,x].astype('uint16')
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                x_match,cor_val,subpix = bcc_pix(Gi,y,n, xLim, res_stack2, offset, offset)

                pos_remove, remove_flag, entry_flag = compare_cor(res_y,
                                                                  [x,x_match, cor_val, subpix, y], threshc)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, subpix, y])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, subpix, y])
        
        rect_res.append(res_y)
      
    #Compare the found right side and compare to left side, and see if it matches. If not, discard.
    res2 = []
    
    a1,a2,cor0 = unpack_rect_res(rect_res)
    
    for val in tqdm(range(len(a1))):
        y_int = a2[val][0]
        x_int = a2[val][1]
        Gi = imgs2a[:,y_int,x_int]
        x_match,cor_val,subpix = bcc_pix(Gi,y_int,n, xLim, imgs1a, offset, offset)
        if(x_match > a1[val][1] - 2 or x_match < a1[val][1] + 2):
            res2.append(a1[val])
    print(len(res2))

    hL_inv = np.linalg.inv(H1)
    hR_inv = np.linalg.inv(H2)
    ptsL = []
    ptsR = []
    for a in range(len(rect_res)):
        b = rect_res[a]
        for q in b:
            xL = q[0]
            y = q[4]
            xR = q[1]
            subx = q[3][1]
            suby = q[3][0]
            
            xL_u = (hL_inv[0,0]*xL + hL_inv[0,1] * (y+suby) + hL_inv[0,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * (y+suby)  + hL_inv[2,2])
            yL_u = (hL_inv[1,0]*xL + hL_inv[1,1] * (y+suby)  + hL_inv[1,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * (y+suby)  + hL_inv[2,2])
            xR_u = (hR_inv[0,0]*(xR+subx) + hR_inv[0,1] * (y+suby)  + hR_inv[0,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * (y+suby)  + hR_inv[2,2])
            yR_u = (hR_inv[1,0]*(xR+subx) + hR_inv[1,1] * (y+suby)  + hR_inv[1,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * (y+suby)  + hR_inv[2,2])
            ptsL.append([xL_u,yL_u])
            ptsR.append([xR_u,yR_u])
            
             
    col_ptsL = np.around(ptsL,0).astype('uint16')
    col_ptsR = np.around(ptsR,0).astype('uint16')
    unique, counts = np.unique(cor0, return_counts=True)
    print(unique)
    print(counts)
    col_arr = scr.get_color(col_refL, col_refR, col_ptsL, col_ptsR)
    cor_arr = scr.create_colcor_arr(cor0)
    tri_res = scr.triangulate_list(ptsL,ptsR, r, t, kL, kR)
    scr.convert_np_ply(np.asarray(tri_res), cor_arr,'test_lookbackb.ply')

def check_pts_ncc(Gi,Gt,n):
    agi = np.sum(Gi)/n
    
    val_i = np.sum((Gi-agi)**2) 
    agt = np.sum(Gt)/n        
    val_t = np.sum((Gt-agt)**2)
    cor = np.sum((Gi-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))
    return cor


def get_neighbor_mods(n_list):
    res = list(itt.combinations(n_list, 2))
    return res

def test_ncc_lookback():
    #load images
    imgFolder = './test_data/testset1/bulb/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgs1,imgs2 = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    col_refL, col_refR = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd,colorIm = True)
    #apply filter
    thresh1 = 10
    imgs1 = np.asarray(scr.mask_inten_list(imgs1,thresh1))
    imgs2 = np.asarray(scr.mask_inten_list(imgs2,thresh1))
    imshape = imgs1[0].shape
    n =8

    imgs1a = np.zeros((n,imshape[0],imshape[1]))
    imgs2a = np.zeros((n,imshape[0],imshape[1]))
    for a in range(n):
        imgs1a[a,:,:]  = imgs1[a,:,:]
        imgs2a[a,:,:] = imgs2[a,:,:]
    
    print(imshape)
    #load matrices
    mat_folder = './test_data/testset1/matrices/'
    kL, kR, r, t = scr.load_mats(mat_folder) 
    f = np.loadtxt(mat_folder + 'f.txt', delimiter = ' ', skiprows = 2)
    #rectify images
    v,w, H1, H2 = scr.rectify_pair(imgs1[0], imgs2[0], f)
    imgs1a,imgs2a = scr.rectify_lists(imgs1,imgs2,f)
    imgs1a = np.asarray(imgs1a)
    imgs2a = np.asarray(imgs2a)
    plt.imshow(imgs1a[0])
    plt.show()
    print(imgs1a.shape)
    #run correlation search on stacks
    offset = 10
    rect_res = []
    xLim = imshape[1]
    yLim = imshape[0]
    threshc = 0.9
    #Take left and compare to right side to find matches
    for y in tqdm(range(offset, yLim-offset)):
        res_y = []
        for x in range(offset, xLim-offset):
            Gi = imgs1a[:,y,x]
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                x_match,cor_val,subpix = ncc.cor_acc_rbf(Gi,y,n, xLim, imgs2a, offset, offset)
                
                pos_remove, remove_flag, entry_flag = compare_cor(res_y,
                                                                  [x,x_match, cor_val, subpix, y], threshc)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, subpix, y])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, subpix, y])
        
        rect_res.append(res_y)
      
    res2 = []
    
    a1,a2,cor0 = unpack_rect_res(rect_res)
    '''
    for val in tqdm(range(len(a1))):
        y_int = int(a2[val][0])
        x_int = int(a2[val][1])
        Gi = imgs2a[:,y_int,x_int]
        x_match,cor_val,subpix = ncc_pix(Gi,y_int,n, xLim, imgs1a, offset, offset)
        if(x_match > a1[val][1] - 2 or x_match < a1[val][1] + 2):
            res2.append(a1[val])
    print(len(res2))
   '''
    hL_inv = np.linalg.inv(H1)
    hR_inv = np.linalg.inv(H2)
    ptsL = []
    ptsR = []
    for a in range(len(rect_res)):
        b = rect_res[a]
        for q in b:
            xL = q[0]
            y = q[4]
            xR = q[1]
            subx = q[3][1]
            suby = q[3][0]
            
            xL_u = (hL_inv[0,0]*xL + hL_inv[0,1] * y + hL_inv[0,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * y  + hL_inv[2,2])
            yL_u = (hL_inv[1,0]*xL + hL_inv[1,1] * y  + hL_inv[1,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * y  + hL_inv[2,2])
            xR_u = (hR_inv[0,0]*(xR+subx) + hR_inv[0,1] * (y+suby)  + hR_inv[0,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * (y+suby)  + hR_inv[2,2])
            yR_u = (hR_inv[1,0]*(xR+subx) + hR_inv[1,1] * (y+suby)  + hR_inv[1,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * (y+suby)  + hR_inv[2,2])
            ptsL.append([xL_u,yL_u])
            ptsR.append([xR_u,yR_u])
            
             
    col_ptsL = np.around(ptsL,0).astype('uint16')
    col_ptsR = np.around(ptsR,0).astype('uint16')
      
    col_arr = scr.get_color(col_refL, col_refR, col_ptsL, col_ptsR)
    tri_res = scr.triangulate_list(ptsL,ptsR, r, t, kL, kR)
    cor_arr = scr.create_colcor_arr(cor0)
    print(np.average(cor0))
    
    scr.convert_np_ply(np.asarray(tri_res), cor_arr,'test_lookback.ply')
    

def comp_corm():
    #load matrices
    mat_folder = './test_data/testset1/matrices/'
    kL, kR, r, t = scr.load_mats(mat_folder) 
    f = np.loadtxt(mat_folder + 'f.txt', delimiter = ' ', skiprows = 2)
    #load images
    imgFolder = './test_data/testset1/bulb/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgs1,imgs2 = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    col_refL, col_refR = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd,colorIm = True)
    n= imgs1.shape[0]
    #apply filter
    thresh1 = 10
    imgs1 = scr.mask_inten_list(imgs1,thresh1)
    imgs2 = scr.mask_inten_list(imgs2,thresh1)
    imshape = imgs1[0].shape

    #rectify images
   
    imgs1,imgs2 = scr.rectify_lists(imgs1,imgs2,f)
    
    #take smaller amount of images for testing ncc
    n = 8
    
    imgs1a = np.zeros((n,imshape[0],imshape[1]))
    imgs2a = np.zeros((n,imshape[0],imshape[1]))
    for a in range(n):
        imgs1a[a]  = imgs1[a]
        imgs2a[a] = imgs2[a]
    #binary conversion of same number of inputs as ncc for bcc
    comN = 4
    res_stack1 = biconv1(imgs1,n,comN)
    res_stack2 = biconv1(imgs2,n,comN)
    
    
    
        
    
    x = 573
    y = 299
    offset = 10

    xLim = imshape[1]

    Gi = imgs1a[:,y,x]
    x_match,cor_val,subpix = ncc_pix(Gi,y,n, xLim, imgs2a, offset, offset)
    Gi2 = res_stack1[:,y,x]
    x_match2,cor_val2,subpix2 = bcc_pix(Gi2,y,n, xLim, res_stack2, offset, offset)
    print('X MATCH')
    print('NCC: ' + str(x_match))
    print('BCC: ' + str(x_match2))
    print('COR VAL')
    print('NCC: ' + str( cor_val))
    print('BCC: ' + str(cor_val2))
    
    print(subpix)
    print(subpix2)

#comp_corm()
test_bcc_lookback()