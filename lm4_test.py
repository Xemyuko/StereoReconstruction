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

def spat_extract(imgs):
    #input:image list
    #output: image stack of neighboring features to each point
    offset = 2
    n=9
    imshape = imgs[0].shape
    res = np.zeros((n*len(imgs),imshape[0],imshape[1]), dtype = imgs[0].dtype)
    for img in imgs:
        for i in range(offset,imshape[0] - offset):
            for j in range(offset,imshape[1] - offset):
                #assign central pixel
                res[0,i,j] = img[i,j]
                #assign cardinal directions, first layer, NSEW
                res[1,i,j] = img[i-1,j]
                res[2,i,j] = img[i+1,j]
                res[3,i,j] = img[i,j-1]
                res[4,i,j] = img[i,j+1]
                #first layer diagonals
                res[5,i,j] = img[i-1,j-1]
                res[6,i,j] = img[i+1,j+1]
                res[7,i,j] = img[i+1,j-1]
                res[8,i,j] = img[i-1,j+1]

    return res

def biconv(imgs):
    pass

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

def ncc_pix():
    pass


def run_test1():
    pass
    #load images
    imgFolder = './test_data/testset1/bulb-multi/b1/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgs1,imgs2 = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    col_refL, col_refR = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd,colorIm = True)
    #pull first 4
    imshape = imgs1[0].shape
    n = 4
    imgs1a = np.zeros((n,imshape[0],imshape[1]))
    imgs2a = np.zeros((n,imshape[0],imshape[1]))
    col_refLa = np.zeros((n,imshape[0],imshape[1]))
    col_refRa = np.zeros((n,imshape[0],imshape[1]))
    for a in range(n):
        imgs1a[a,:,:]  = imgs1[a,:,:]
        imgs2a[a,:,:]  = imgs2[a,:,:]
        col_refLa[a,:,:] = col_refL[a,:,:]
        col_refRa[a,:,:] = col_refR[a,:,:]
    #apply filter
    thresh1 = 30
    imgs1 = np.asarray(scr.mask_inten_list(imgs1a,thresh1))
    imgs2 = np.asarray(scr.mask_inten_list(imgs2a,thresh1))
    #load matrices
    mat_folder = './test_data/testset1/matrices/'
    kL, kR, r, t = scr.load_mats(mat_folder) 
    f = np.loadtxt(mat_folder + 'f.txt', delimiter = ' ', skiprows = 2)
    #rectify images
    v,w, H1, H2 = scr.rectify_pair(imgs1[0], imgs2[0], f)
    imgs1,imgs2 = scr.rectify_lists(imgs1,imgs2,f)
    #apply spatextract
    imgs1 = spat_extract(imgs1)
    imgs2 = spat_extract(imgs2)
    
    cor_thresh = 0.0
    
    