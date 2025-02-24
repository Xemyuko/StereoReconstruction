# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:13:29 2025

@author: Admin
"""

import numpy as np
import scripts as scr
import matplotlib.pyplot as plt
import ncc_core as ncc
from tqdm import tqdm
import cv2
import json
import numba
import time 
import threading as thr
from numba import cuda as cu
import time
import os

import confighandler as chand
import scipy.signal as sig
from scipy.interpolate import Rbf
import scipy.interpolate
import scipy.linalg as sclin
import tkinter as tk
import inspect
import csv
import bcc_core as bcc
import itertools as itt


#used for comparing floating point numbers to avoid numerical errors
float_epsilon = 1e-9
def grad_assign1(centval, v1,v2,thresh):
    if abs(centval - v1) < thresh: #0,3,4
        if abs(v1 -v2) < thresh:
            return 0
        elif v1-v2 > 0:
            return 4
        else:
            return 3
    elif centval - v1 < 0: #1,5
        if abs(v1-v2) < thresh:
            return 1 
        else:
            return 5
    else: #2,6
        if abs(v1-v2) < thresh:
            return 2
        else:
            return 6
def grad_assign2(centval, v1,v2,v3,thresh):
    if abs(centval-v1) < thresh: 
        if abs(v1 -v2) < thresh: 
            if abs(v2-v3) < thresh:
                return 0
            elif v2-v3 < 0:
                return 1
            else:
                return 2
        elif v1 -v2 > 0: 
            if abs(v2-v3) < thresh:
                return 3
            elif v2-v3 < 0:
                return 4
            else:
                return 5
        else: 
            if abs(v2-v3) < thresh:
                return 6
            elif v2-v3 < 0:
                return 7
            else:
                return 8
    elif centval-v1 > 0: 
        if abs(v1 -v2) < thresh:
            if abs(v2-v3) < thresh:
                return 9
            elif v2-v3 < 0:
                return 10
            else:
                return 11
        elif v1 -v2 > 0: 
            if abs(v2-v3) < thresh:
                return 12
            elif v2-v3 < 0:
                return 13
            else:
                return 14
        else: 
            if abs(v2-v3) < thresh:
                return 15
            elif v2-v3 < 0:
                return 16
            else:
                return 17
    else:
        if abs(v1 -v2) < thresh:
            if abs(v2-v3) < thresh:
                return 18
            elif v2-v3 < 0:
                return 19
            else:
                return 21
        elif v1 -v2 > 0: 
            if abs(v2-v3) < thresh:
                return 22
            elif v2-v3 < 0:
                return 23
            else:
                return 24
        else: 
            if abs(v2-v3) < thresh:
                return 25
            elif v2-v3 < 0:
                return 26
            else:
                return 27
def grad_ext1(imgs, thresh = 5, sclfac = 20):
    n = 7
    imshape = imgs[0].shape
    offset = 5
    res = np.zeros((n*len(imgs),imshape[0],imshape[1]), dtype = imgs[0].dtype)
    imgs = np.asarray(imgs,dtype = 'int16')
    for ind in tqdm(range(len(imgs))):
        img = imgs[ind]
        for i in range(offset,imshape[0] - offset):
            for j in range(offset,imshape[1] - offset):
                #assign central point value
                cent_val = img[i,j] 
                #assign cardinal directions, first layer, NSEW
                res[0,i,j] = grad_assign1(cent_val,img[i-1,j],img[i-2,j],thresh)*sclfac
                res[1,i,j] = grad_assign1(cent_val,img[i+1,j],img[i+2,j],thresh)*sclfac
                res[2,i,j] = grad_assign1(cent_val,img[i,j-1],img[i,j-2],thresh)*sclfac
                res[3,i,j] = grad_assign1(cent_val,img[i,j+1],img[i,j+2],thresh)*sclfac
                
                #first layer diagonals
                
                res[4,i,j] = grad_assign1(cent_val,img[i-1,j-1],img[i-2,j-2],thresh)*sclfac
                res[5,i,j] = grad_assign1(cent_val,img[i+1,j+1],img[i+2,j+2],thresh)*sclfac
                res[6,i,j] = grad_assign1(cent_val,img[i+1,j-1],img[i+2,j-2],thresh)*sclfac
                res[7,i,j] = grad_assign1(cent_val,img[i-1,j+1],img[i-2,j+2],thresh)*sclfac
                    
    return res

def grad_ext2(imgs,thresh = 5, sclfac = 5):
    n = 7
    imshape = imgs[0].shape
    offset = 5
    res = np.zeros((n*len(imgs),imshape[0],imshape[1]), dtype = imgs[0].dtype)
    imgs = np.asarray(imgs,dtype = 'int16')
    for ind in tqdm(range(len(imgs))):
        img = imgs[ind]
        for i in range(offset,imshape[0] - offset):
            for j in range(offset,imshape[1] - offset):
                #assign central point value
                cent_val = img[i,j] 
                
    return res
def grad_ext3(imgs): 
    n = 7
    imshape = imgs[0].shape
    offset = 5
    res = np.zeros((n*len(imgs),imshape[0],imshape[1]), dtype = imgs[0].dtype)
    imgs = np.asarray(imgs,dtype = 'int16')
    for ind in tqdm(range(len(imgs))):
        img = imgs[ind]
        for i in range(offset,imshape[0] - offset):
            for j in range(offset,imshape[1] - offset):
                #assign central point value
                cent_val = img[i,j]
                #assign cardinal directions, first layer, NSEW
                res[0,i,j] = np.abs(cent_val-img[i-1,j]) + np.abs(img[i-1,j]-img[i-2,j])
                res[1,i,j] = np.abs(cent_val-img[i+1,j]) + np.abs(img[i+1,j]-img[i+2,j])
                res[2,i,j] = np.abs(cent_val-img[i,j-1]) + np.abs(img[i,j-1]-img[i,j-2])
                res[3,i,j] = np.abs(cent_val-img[i,j+1]) + np.abs(img[i,j+1]-img[i,j+2])
                #first layer diagonals
                res[4,i,j] = np.abs(cent_val-img[i-1,j-1]) + np.abs(img[i-1,j-1]-img[i-2,j-2])
                res[5,i,j] = np.abs(cent_val-img[i+1,j+1]) + np.abs(img[i+1,j+1]-img[i+2,j+2])
                res[6,i,j] = np.abs(cent_val-img[i+1,j-1]) + np.abs(img[i+1,j-1]-img[i+2,j-2])
                res[7,i,j] = np.abs(cent_val-img[i-1,j+1]) + np.abs(img[i-1,j+1]-img[i-2,j+2])
                
    return res

def grad_ext4(imgs,thresh = 5, sclfac = 5):
    n = 1
    x_vals = np.linspace(-n,n,2*n+1).astype('int8')
    y_vals = np.linspace(-n,n,2*n+1).astype('int8')
    n_list = []
    for i in x_vals:
        for j in y_vals:
            ent = [j,i]
            n_list.append(ent)
    n2 = len(n_list)
    imshape = imgs[0].shape
    offset = 5
    res = np.zeros((n2*len(imgs),imshape[0],imshape[1]), dtype = imgs[0].dtype)
    imgs = np.asarray(imgs,dtype = 'int16')
    for ind in tqdm(range(len(imgs))):
        img = imgs[ind]
        for i in range(offset,imshape[0] - offset):
            for j in range(offset,imshape[1] - offset):
               #assign central point value
               cent_val = img[i,j] 
               #assign cardinal directions, first layer, NSEW
               for e in range(n2):
                   res[e,i,j] = grad_assign2(cent_val,img[i+n_list[e][0],j+n_list[e][1]],img[i+n_list[e][0]*2,j+n_list[e][1]]*2,img[i+n_list[e][0]*3,j+n_list[e][1]*3],thresh)*sclfac
               
              
         
                
def spat_ext(imgs, n = 3):
    #input:image list
    #output: image stack of neighboring features to each point
    x_vals = np.linspace(-n,n,2*n+1).astype('int8')
    y_vals = np.linspace(-n,n,2*n+1).astype('int8')
    n_list = []
    for i in x_vals:
        for j in y_vals:
            ent = [j,i]
            n_list.append(ent)
    n2=len(n_list)
    imshape = imgs[0].shape
    res = np.zeros((n2*len(imgs),imshape[0],imshape[1]), dtype = imgs[0].dtype)
    for b in tqdm(range(len(imgs))):
        img = imgs[b]
        for i in range(n,imshape[0] - n):
            for j in range(n,imshape[1] - n):
                for a in range(n2):
                    res[a,i,j] = img[i+n_list[a][0],j+n_list[a][1]]
    return res
def comb_ext(imgs):
    #print('SP')
    #sp = spat_ext(imgs, n = 3)
    print('GR')
    gr = grad_ext1(imgs)
    #print('GR-Mag')
    #gr3 = grad_ext3(imgs)
    res = np.concatenate((imgs,gr))
    return res
def biconv1(imgs):
    n = len(imgs)
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

def comcor1(res_list, entry_val, threshold):
    remove_flag = False
    pos_remove = 0
    entry_flag = False
    counter = 0

    if(entry_val[1] < 0 or entry_val[2] < threshold):

        return pos_remove,remove_flag,entry_flag
    for i in range(len(res_list)):       
        
        if(res_list[i][1] == entry_val[1] and res_list[i][3][0] - entry_val[3][0] < float_epsilon and
           res_list[i][3][1] - entry_val[3][1] < float_epsilon):
            #duplicate found, check correlation values and mark index for removal
            remove_flag = (res_list[i][2] < entry_val[2])
            
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
    max_mod = [0,0]
    
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = maskR[:,y,xi]
        chkres = Gi-Gt
        chk = np.sum(chkres == 0)/n
        if(chk > max_cor):
            max_index = xi
            max_cor = chk
         
    Gup = maskR[:,y-1, max_index]
    chkres = Gi-Gup
    chk = np.sum(chkres == 0)/n
    if(chk > max_cor):
        max_mod  = [-1,0]
        max_cor = chk
    
    Gdn = maskR[:,y+1, max_index]
    chkres = Gi-Gdn
    chk = np.sum(chkres == 0)/n
    if(chk > max_cor):
        max_mod  = [1,0]
        max_cor = chk
        
    return max_index,max_cor,max_mod
def col_depth(pts):
    #get depth range
    z_ext = pts[:,2]
    
    z_range = np.linspace(np.min(z_ext),np.max(z_ext),60)
    
def run_test1():
    #load images
    #imgFolder = './test_data/testset1/bulb4lim/'
    imgFolder = './test_data/testset1/bulb-multi/b6/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgs1,imgs2 = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    col_refL, col_refR = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd,colorIm = True)
    imshape = imgs1[0].shape
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
    
    imgs1 = comb_ext(imgs1)
    imgs2 = comb_ext(imgs2)


    n2 = len(imgs1)
    print('TOTAL INPUT: ' + str(n2))
    cor_thresh = 0.9
    offset = 10
    rect_res = []
    xLim = imshape[1]
    yLim = imshape[0]

    for y in tqdm(range(offset, yLim-offset)):
        res_y = []
        for x in range(offset, xLim-offset):
            Gi = imgs1[:,y,x].astype('uint8')
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                x_match,cor_val,subpix= ncc_pix(Gi,y,n2, xLim, imgs2, offset, offset)
                pos_remove, remove_flag, entry_flag = comcor1(res_y,
                                                                  [x,x_match, cor_val, subpix, y], cor_thresh)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, subpix, y])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, subpix, y])
        
        rect_res.append(res_y)
    
    cor_list = []
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
            cor_list.append(q[2])
          

        
    col_ptsL = np.around(ptsL,0).astype('uint16')
    col_ptsR = np.around(ptsR,0).astype('uint16')  
    print(np.min(cor_list))
    print(np.max(cor_list))
    col_arr = scr.get_color(col_refL, col_refR, col_ptsL, col_ptsR)      
    #col_arr = scr.create_colcor_arr(cor_list, cor_thresh)
    tri_res = scr.triangulate_list(ptsL,ptsR, r, t, kL, kR)
    col_depth(tri_res)  
    #scr.convert_np_ply(np.asarray(tri_res), col_arr,"bulbcomb-7ncc.ply")


run_test1()
