# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:56:43 2024

@author: mg
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
import matplotlib.pyplot as plt
import confighandler as chand
from scipy.interpolate import Rbf
import scipy.interpolate
import scipy.linalg as sclin
import tkinter as tk
import inspect



def triangulate(pt1,pt2,R,t,kL,kR):
    #Create calc matrices 
    Al = np.c_[kL, np.asarray([[0],[0],[0]])]
    

    RT = np.c_[R, t]

    Ar = kR @ RT

    sol0 = pt1[1] * Al[2,:] - Al[1,:]
    sol1 = -pt1[0] * Al[2,:] + Al[0,:]
    sol2 = pt2[1] * Ar[2,:] - Ar[1,:]
    sol3 = -pt2[0] * Ar[2,:] + Ar[0,:]

    solMat = np.stack((sol0,sol1,sol2,sol3))
    #Apply SVD to solution matrix to find triangulation
    U,s,vh = np.linalg.svd(solMat,full_matrices = True)

    Q = vh[3,:]

    Q /= Q[3]
    return Q[0:3]

def triangulate2(pt1,pt2,R,t,kL,kR):
    #Create calc matrices 
    
    Al = np.c_[kL, np.asarray([[0],[0],[0]])]
    

    RT = np.c_[R, t]

    Ar = kR @ RT
    sol0 = pt1[1] * Al[2,:] - Al[1,:]
    sol1 = -pt1[0] * Al[2,:] + Al[0,:]
    sol2 = pt2[1] * Ar[2,:] - Ar[1,:]
    sol3 = -pt2[0] * Ar[2,:] + Ar[0,:]
    
    solMat = np.stack((sol0,sol1,sol2,sol3))
    solMat2 = solMat.T@solMat
    #Apply SVD to solution matrix to find triangulation
    U,s,vh = np.linalg.svd(solMat2,full_matrices = True)
    Q = vh[3,:]

    Q /= Q[3]

    return Q[0:3]

def tri_check():
    data_folder = './test_data/testset0/240312_angel/'
    #data_folder = './test_data/testset0/240411_hand0/'
    #load pcf of known good data
    ref_file = '000POS000Rekonstruktion030.pcf'
    #ref_file = 'pws/000POS000Rekonstruktion030.pcf'
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(data_folder + ref_file)
    mat_folder = './test_data/testset0/matrices/'
    kL, kR, R, t = scr.load_mats(mat_folder)
    #triangulate known good points
   
                   
    inspect_ind =0
    
    p1 = xy1[inspect_ind]
    p2 = xy2[inspect_ind]
    res1 = triangulate(p1,p2,R,t, kL, kR)
    res2 = triangulate2(p1,p2,R,t, kL, kR)
    res3 = geom_arr[inspect_ind]
    print("R:")
    print(R)
    print("t:")
    print(t)
    print("kL:")
    print(kL)
    print("kR:")
    print(kR)
    print("Input Point Pair:")
    print(p1)
    print(p2)
    print("######")
    print("Current Triangulation:")
    print(res1)
    print("Test Triangulation:")
    print(res2)
    print('#################')
    print("Reference Triangulation")
    print(res3)
    print('#################')
    print("Error Current:")
    print(res1/res3)
    print("#################")
    print("Error Test:")
    print(res2/res3)
    print("#################")

    full_check = False
    diff_chk1 = []
    diff_chk2 = []
    res_tri1 = []
    res_tri2 = []
    avg_diff1 = np.asarray([0.0,0.0,0.0])
    avg_diff2 = np.asarray([0.0,0.0,0.0])
    if full_check:
        for i in tqdm(range(len(xy1))):
            p1 = xy1[i]
            p2 = xy2[i]
            res1 = triangulate(p1,p2,R,t, kL, kR)
            
            res_tri1.append(res1)
            
            res3 = geom_arr[i]
            diff_chk1.append(res1/res3)
            
            avg_diff1+=res1/res3
            
        for i in tqdm(range(len(xy1))):
            p1 = xy1[i]
            p2 = xy2[i]
            res2 = triangulate2(p1,p2,R,t,kL,kR)
            res_tri2.append(res2)
            res3 = geom_arr[i]
            diff_chk2.append(res2/res3)
            avg_diff2+=res2/res3
        avg_diff1/=len(xy1)
        print('\n')
        print('Average Error Current:')
        print(avg_diff1)
        avg_diff2/=len(xy1)
        print('Average Error Test:')
        print(avg_diff2)

tri_check()

