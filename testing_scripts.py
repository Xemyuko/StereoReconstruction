# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:09:26 2023

@author: Admin
"""

import numpy as np
import scripts as scr
import scipy.linalg as sclin
def t1():
    #define data sources
    folder_statue = "./test_data/statue/"
    matrix_folder = "matrix_folder/"
    left_folder = "camera_L/"
    right_folder = "camera_R/"
    input_data = "Rekonstruktion30.pcf"
    #load data
    kL, kR, R, t = scr.initial_load(1,folder_statue + matrix_folder)
    imgL,imgR = scr.load_images(folderL = folder_statue+left_folder, folderR = folder_statue+right_folder)
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(folder_statue + input_data)
    #F = scr.find_f_mat(imgL[0],imgR[0])
    kL_inv = np.linalg.inv(kL)
    kR_inv = np.linalg.inv(kR)
    
    ind = 59
    
    pL = xy1[ind]
    pR = xy2[ind]
    print('P________')
    print(pL)
    print(pR)

    projR = np.append(R,t,axis = 1) 
    projL = np.append(np.identity(3),np.asarray([[0],[0],[0]]),axis = 1)
    A_L = kL @ projL
    A_R = kR @ projR
    r0 = pL[1]*A_L[2,:] - A_L[1,:]
    r1 = A_L[0,:] - pL[0]*A_L[2,:]
    r2 = pR[1]*A_R[2,:] - A_R[1,:]
    r3 = A_R[0,:] - pR[0]*A_R[2,:]
    somat = np.vstack((r0,r1,r2,r3))
    print('S________')
    print(somat)
    
    sts = somat.T @ somat
    
    u, s, vh = np.linalg.svd(sts, full_matrices = True)
    Q = vh[:,3]
    
    Q *= 1/Q[3]
    Q = Q[:3]
    
    print('Q________')
    print(Q)
    print('A________')
    print(geom_arr[ind])

t1()