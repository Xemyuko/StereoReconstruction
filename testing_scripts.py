# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:09:26 2023

@author: Admin
"""

import numpy as np
import scripts as scr
import scipy.linalg as sclin
import ncc_core as ncc
import confighandler as ch
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
def distance3D(pt1,pt2):
    res = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)
    return res
def t2():
    #set reference pcf file, folders for images and matrices. 
    folder_statue = "./test_data/statue/"
    matrix_folder = "matrix_folder/"
    left_folder = "camera_L/"
    right_folder = "camera_R/"
    input_data = "Rekonstruktion30.pcf"
    #known correct tmod
    t_mod = 0.416657633
    #load reference data
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(folder_statue + input_data)
    #create config object
    config = ch.ConfigHandler()
    #assign config values
    config.mat_folder = folder_statue + matrix_folder
    config.tmod = 1.0
    config.speed_mode = 1
    config.left_folder = folder_statue + left_folder
    config.right_folder = folder_statue + right_folder
    config.precise = 1
    config.corr_map_out = 0
    #use internal correlation
    ptsL,ptsR = ncc.cor_internal(config)
    #rng comparison point indices
    comp_num = 20
    rng = np.random.default_rng()
    indA1 = rng.integers(len(ptsL),size =comp_num)
    indA2 = rng.integers(len(ptsL),size =comp_num)
    #pull points out from ptsL and ptsR
    ptA1 = []
    ptA2 = []
    for i in indA1:
        ptA1.append([ptsL[i], ptsR[i]])
    for i in indA2:
        ptA2.append([ptsL[i],ptsR[i]])
    #find matching points to get comparison points
    #loop through first point pairs
    matched_inds1 = []
    for test_points in ptA1:
        '''
        inLx = i[0][0]
        inLy = i[0][1]
        inRx = i[1][0]
        inRy = i[1][1]
        '''
        diff1 = 100000.0
        match1Ind = 0
        #loop through paired reference points
        for ind in range(len(xy1)):
            refPointL = xy1[ind]
            refPointR = xy2[ind]
            #compare first set
            difL = test_points[0] - refPointL
            difR = test_points[1] - refPointR
            #sum of squared differences
            diff_comp = np.sum(difL - difR)**2
            if diff_comp < diff1:
                diff1 = diff_comp
                match1Ind = ind
        matched_inds1.append(match1Ind)
    matched_inds2 = []
    #loop through second point pairs
    for i in ptA2:
        diff2 = 100000.0
        match2Ind = 0
        for ind in range(len(xy1)):
            refPointL = xy1[ind]
            refPointR = xy2[ind]
            
            #compare first set
            difL = test_points[0] - refPointL
            difR = test_points[1] - refPointR
            #sum of squared differences
            diff_comp = np.sum(difL - difR)**2
            if diff_comp < diff2:
                diff2 = diff_comp
                match2Ind = ind
        matched_inds2.append(match2Ind)
    
    #retrieve found matching reference points
    ptB1 = []
    
    ptB2 = []
    for i in matched_inds1:
        ptB1.append([xy1[i], xy2[i], geom_arr[i]])
    for i in matched_inds2:
        ptB2.append([xy1[i], xy2[i], geom_arr[i]])
    #check first comparison values
    print(ptB1[0])
    print("______________________________________")
    print(ptA1[0])
    #calculate distances between reference points
    ref_dist = []
    for i in range(len(ptB1)):
        dist = distance3D(ptB1[2],ptB2[2])
        ref_dist.append(dist)
    
    #create loop for testing incremental values of t-vector scaling factor
    kL, kR, r_vec, t_vec = scr.initial_load(1.0,folder_statue +matrix_folder)
    kL_inv = np.linalg.inv(kL)
    kR_inv = np.linalg.inv(kR)
    inc = 0.001
    max_tmod = 1
    tmod_start = inc
    #create new set of point pairs
    while tmod_start < max_tmod:
        t_vec *= tmod_start
        tri_res = scr.triangulate_list(ptsL,ptsR, r_vec, t_vec, kL_inv, kR_inv, config.precise)
        
t2()