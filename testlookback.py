# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:41:06 2025

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



def test_ncc_lookback():
    #load images
    imgFolder = './test_data/testset1/bulb/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgs1,imgs2 = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    col_refL, col_refR = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd,colorIm = True)
    n= imgs1.shape[0]
    #apply filter
    thresh1 = 30
    imgs1 = np.asarray(scr.mask_inten_list(imgs1,thresh1))
    imgs2 = np.asarray(scr.mask_inten_list(imgs2,thresh1))
    imshape = imgs1[0].shape
    
    
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

    #run correlation search on stacks
    offset = 1
    rect_res = []
    xLim = imshape[1]
    yLim = imshape[0]
    #Take left and compare to right side to find matches
    for y in tqdm(range(offset, yLim-offset)):
        res_y = []
        for x in range(offset, xLim-offset):
            Gi = imgs1a[:,y,x].astype('int8')
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                x_match,cor_val,subpix = ncc.cor_acc_pix(Gi,y,n, xLim, imgs2a, offset, offset)

                pos_remove, remove_flag, entry_flag = ncc.compare_cor(res_y,
                                                                  [x,x_match, cor_val, subpix, y], 0.9)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, subpix, y])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, subpix, y])
        
        rect_res.append(res_y)
    '''
    #Compare the found right side and compare to left side, and see if it matches. If not, discard.
    rect_res2 = []
    for entvb in rect_res:
        ent2 = []
        for ent in entvb:
            Gi2 = imgs2a[:,int(ent[4]),int(ent[0])]
            x_match2, cor_val2, subpix2 = ncc.cor_acc_pix(Gi2,ent[4],n,xLim,imgs1a,offset,offset)
            if(x_match2+subpix2[1] == ent[1]+ent[3][1]):
                ent2.append(ent)
        rect_res2.append(ent2)
            
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
            
            xL_u = (hL_inv[0,0]*xL + hL_inv[0,1] * (y+suby) + hL_inv[0,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * (y+suby)  + hL_inv[2,2])
            yL_u = (hL_inv[1,0]*xL + hL_inv[1,1] * (y+suby)  + hL_inv[1,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * (y+suby)  + hL_inv[2,2])
            xR_u = (hR_inv[0,0]*(xR+subx) + hR_inv[0,1] * (y+suby)  + hR_inv[0,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * (y+suby)  + hR_inv[2,2])
            yR_u = (hR_inv[1,0]*(xR+subx) + hR_inv[1,1] * (y+suby)  + hR_inv[1,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * (y+suby)  + hR_inv[2,2])
            ptsL.append([xL_u,yL_u])
            ptsR.append([xR_u,yR_u])
            
             
    col_ptsL = np.around(ptsL,0).astype('uint16')
    col_ptsR = np.around(ptsR,0).astype('uint16')
      
    col_arr = scr.get_color(col_refL, col_refR, col_ptsL, col_ptsR)
    tri_res = scr.triangulate_list(ptsL,ptsR, r, t, kL, kR)
    scr.convert_np_ply(np.asarray(tri_res), col_arr,'test_lookback.ply')
    
test_ncc_lookback()