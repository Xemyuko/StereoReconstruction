# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:09:26 2023

@author: Admin
"""

import numpy as np
import scripts as scr
import matplotlib.pyplot as plt
import ncc_core as ncc
import confighandler as ch
from tqdm import tqdm
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
    config.precise = 0
    config.corr_map_out = 0
    #use internal correlation
    ptsL,ptsR = ncc.cor_internal(config)
    #rng comparison point indices
    comp_num = 10
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
            diff_comp = np.sum(difL**2 + difR**2)
            if diff_comp < diff1:
                diff1 = diff_comp
                match1Ind = ind
        matched_inds1.append(match1Ind)
    matched_inds2 = []
    #loop through second point pairs
    for test_points in ptA2:
        diff2 = 100000.0
        match2Ind = 0
        for ind in range(len(xy1)):
            refPointL = xy1[ind]
            refPointR = xy2[ind]
            
            #compare first set
            difL = test_points[0] - refPointL
            difR = test_points[1] - refPointR
            #sum of squared differences
            diff_comp = np.sum(difL**2 + difR**2)
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
        
    #check first values
    print("Point Pair 1")
    print("Found:")
    print(ptA1[0])
    print("______________________________________")
    print("Matched Reference:")
    print(ptB1[0])
    print("______________________________________")
    print("Point Pair 2")
    print("Found:")
    print(ptA2[0])
    print("______________________________________")
    print("Matched Reference:")
    print(ptB2[0])
    
    
    
    #calculate distances between reference points
    ref_dist = []
    for i in range(len(ptB1)):
        dist = distance3D(ptB1[i][2],ptB2[i][2])
        ref_dist.append(dist)
    #Need to match relative distance scales    
    #Calculate average of distances in reference points
    ref_dist_avg = np.average(np.asarray(ref_dist))
    #Calculate triangulation of test pointswith tmod = 1
    kL, kR, r_vec, t_vec = scr.initial_load(1.0,folder_statue + matrix_folder)
    kL_inv = np.linalg.inv(kL)
    kR_inv = np.linalg.inv(kR)
    #create new set of point pairs for triangulation
    ptsLCheck1 = []
    ptsRCheck1 = []
    ptsLCheck2 = []
    ptsRCheck2 = []
    for i,j in zip(ptA1,ptA2):
        ptsLCheck1.append(i[0])
        ptsRCheck1.append(i[1])
        ptsLCheck2.append(j[0])
        ptsRCheck2.append(j[1])
    dist_scale_tri = scr.triangulate_list_nobar(ptsLCheck1,ptsRCheck1, r_vec, t_vec, kL_inv, kR_inv, config.precise)
    #Calculate average of distances in test points
    dist_scale_avg_test = np.average(np.asarray(dist_scale_tri))
    #Divide the averages by each other to find a relative distance scale factor
    rel_dist_scale = dist_scale_avg_test/ref_dist_avg
    #loop through incremental values of t-vector scaling factor,
    # apply distance scale factor, and compare to reference 

    #create loop for testing incremental values of t-vector scaling factor
    inc = 0.001
    max_tmod = 1
    tmod_start = inc
    
    #loop triangulation with varying scale factor  
    dist_score = 10**9
    tmod_res = 0
    while tmod_start < max_tmod:
        t_vec_mod = tmod_start *t_vec
        tri_res1 = scr.triangulate_list_nobar(ptsLCheck1,ptsRCheck1, r_vec, t_vec_mod, kL_inv, kR_inv, config.precise)
        tri_res2 = scr.triangulate_list_nobar(ptsLCheck2,ptsRCheck2, r_vec, t_vec_mod, kL_inv, kR_inv, config.precise)
        #check distances, compare to ref_dist array
        dist_check = []
        for i,j in zip(tri_res1,tri_res2):
            dist_check.append(distance3D(i,j)*rel_dist_scale)

        
        dist_score_check = np.sum((np.asarray(dist_check)- np.asarray(ref_dist))**2)/10**8
        if(dist_score_check < dist_score):
            dist_score = dist_score_check
            tmod_res = tmod_start
        tmod_start+=inc
    print(dist_score)
    print(tmod_res)

def remove_z_outlier(geom_arr, col_arr):
    ind_del = []
    for i in range(geom_arr.shape[0]):

        if geom_arr[i,2] > np.max(geom_arr[:,2])*0.75:
            ind_del.append(i)

    geom_arr = np.delete(geom_arr, np.asarray(ind_del), axis = 0)

    return geom_arr

def t3(): 
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
    #Remove outlier z reference data
    geom_arr=remove_z_outlier(geom_arr, col_arr)
    #create config object
    config = ch.ConfigHandler()
    #assign config values
    config.mat_folder = folder_statue + matrix_folder
    config.tmod = 1.0
    config.speed_mode = 0
    config.left_folder = folder_statue + left_folder
    config.right_folder = folder_statue + right_folder
    config.precise = 1
    config.corr_map_out = 0
    #use internal correlation
    ptsL,ptsR = ncc.cor_internal(config)
    #Calculate extreme values of reference point cloud
    minRefX = np.min(geom_arr[:,0])
    maxRefX = np.max(geom_arr[:,0])
    minRefY = np.min(geom_arr[:,1])
    maxRefY = np.max(geom_arr[:,1])
    minRefZ = np.min(geom_arr[:,2])
    maxRefZ = np.max(geom_arr[:,2])
    refDistX = maxRefX - minRefX
    refDistY = maxRefY - minRefY
    refDistZ = maxRefZ - minRefZ
    print('Ref Dist X: ' + str(refDistX))
    print('Ref Dist Y: ' + str(refDistY))
    print('Ref Dist Z: ' + str(refDistZ))
    print('_______')
    print('Ref Z/X: ' + str(refDistZ/refDistX))
    print('Ref Z/Y: ' + str(refDistZ/refDistY))
    print('_______')
    #Run triangulation on ptsL and ptsR
    kL, kR, r_vec, t_vec = scr.initial_load(1.0,folder_statue + matrix_folder)
    kL_inv = np.linalg.inv(kL)
    kR_inv = np.linalg.inv(kR)
    '''
    #adjust t_Vec to 0.5 for middle of the range
    check_tri_res = scr.triangulate_list_nobar(ptsL,ptsR, r_vec, t_vec*0.5, kL_inv, kR_inv, config.precise)
    check_tri_res = np.asarray(check_tri_res)
    
    #Calculate extreme x and y values of test triangulation
    minCheckX = np.min(check_tri_res[:,0])
    maxCheckX = np.max(check_tri_res[:,0])
    minCheckY = np.min(check_tri_res[:,1])
    maxCheckY = np.max(check_tri_res[:,1])
    minCheckZ = np.min(check_tri_res[:,2])
    maxCheckZ = np.max(check_tri_res[:,2])
    checkDistX = maxCheckX - minCheckX
    checkDistY = maxCheckY - minCheckY
    checkDistZ = maxCheckZ - minCheckZ
    print('Chk Dist X: ' + str(checkDistX))
    print('Chk Dist Y: ' + str(checkDistY))
    print('Chk Dist Z: ' + str(checkDistZ))
    print('________')
    print('Chk Z/X: ' + str(checkDistZ/checkDistX))
    print('Chk Z/Y: ' + str(checkDistZ/checkDistY))
    print('_______')
    print('Ref X/Chk X: ' + str(refDistX/checkDistX))
    print('Ref Y/Chk Y: ' + str(refDistY/checkDistY))
    print('Ref Z/Chk Z: ' + str(refDistZ/checkDistZ))
    print('_________')
    scaleFactor = (refDistX/checkDistX + refDistY/checkDistY)/2
    print('Avg XY Ratio ScaleFactor: ' + str(scaleFactor))
    print('Ref Dist X/ScaleFactor: ' + str(refDistX/scaleFactor))
    print('Ref Dist Y/ScaleFactor: ' + str(refDistY/scaleFactor))
    print('Ref Dist Z/ScaleFactor: ' + str(refDistZ/scaleFactor))
    print('_________')
    #confirm with known good tmod value
    confirm_tri_res = scr.triangulate_list_nobar(ptsL,ptsR, r_vec, t_vec*t_mod, kL_inv, kR_inv, config.precise)
    confirm_tri_res = np.asarray(confirm_tri_res)
    minConfirmX = np.min(confirm_tri_res[:,0])
    maxConfirmX = np.max(confirm_tri_res[:,0])
    minConfirmY = np.min(confirm_tri_res[:,1])
    maxConfirmY = np.max(confirm_tri_res[:,1])
    minConfirmZ = np.min(confirm_tri_res[:,2])
    maxConfirmZ = np.max(confirm_tri_res[:,2])
    confirmDistX = maxConfirmX - minConfirmX
    confirmDistY = maxConfirmY - minConfirmY
    confirmDistZ = maxConfirmZ - minConfirmZ
    print('Cnf Dist X: ' + str(confirmDistX))
    print('Cnf Dist Y: ' + str(confirmDistY))
    print('Cnf Dist Z: ' + str(confirmDistZ))
    print('________')
    print('Cnf Z/X: ' + str(confirmDistZ/confirmDistX))
    print('Cnf Z/Y: ' + str(confirmDistZ/confirmDistY))
    print('_______')
    print('Ref X/Cnf X: ' + str(refDistX/confirmDistX))
    print('Ref Y/Cnf Y: ' + str(refDistY/confirmDistY))
    print('Ref Z/Cnf Z: ' + str(refDistZ/confirmDistZ))
    print('_________')
    scaleFactorCnf = (refDistX/confirmDistX + refDistY/confirmDistY)/2
    print('Avg XY Ratio ScaleFactorCnf: ' + str(scaleFactorCnf))
    print('Ref Dist X/ScaleFactorCnf: ' + str(refDistX/scaleFactorCnf))
    print('Ref Dist Y/ScaleFactorCnf: ' + str(refDistY/scaleFactorCnf))
    print('Ref Dist Z/ScaleFactorCnf: ' + str(refDistZ/scaleFactorCnf))
    print('Scalefactor Comparison (1/cnf): ' + str(scaleFactor/scaleFactorCnf))
    print('_________')
    '''
    #loop through adjusting triangulation by varying the tmod scale factor and comparing the scaled result with the 
    inc = 0.01
    max_tmod = 1.0
    opt_search_score = 100
    opt_tmod = inc
    ref_score_x = refDistZ/refDistX
    ref_score_y = refDistZ/refDistY
    ref_score_a = refDistX/refDistY
    #dist_x_arr = []
    #dist_y_arr = []
    #dist_z_arr = []
    #ratio_zx_arr = []
    #ratio_zy_arr = []
    #scoreA_arr = []
    #scoreB_arr = []
    #scoreC_arr = []
    data_arr = []
    for i in tqdm(np.arange(inc,max_tmod,inc)):
        search_tri_res = scr.triangulate_list_nobar(ptsL,ptsR, r_vec, t_vec*i, kL_inv, kR_inv, config.precise)
        search_tri_res = np.asarray(search_tri_res)
        minSearchX = np.min(search_tri_res[:,0])
        maxSearchX = np.max(search_tri_res[:,0])
        minSearchY = np.min(search_tri_res[:,1])
        maxSearchY = np.max(search_tri_res[:,1])
        minSearchZ = np.min(search_tri_res[:,2]) 
        maxSearchZ = np.max(search_tri_res[:,2]) 
        searchDistX = maxSearchX - minSearchX
        searchDistY = maxSearchY - minSearchY
        searchDistZ = maxSearchZ - minSearchZ
        #search_scoreA = np.abs(searchDistX - refDistX/scaleFactor) + np.abs(searchDistY - refDistY/scaleFactor) +np.abs(searchDistZ - refDistZ/scaleFactor)
        search_score = np.abs(searchDistZ/searchDistX - ref_score_x) + np.abs(searchDistZ/searchDistY - ref_score_y) 
        #search_scoreC = np.abs(searchDistZ/searchDistX - ref_score_x) + np.abs(searchDistZ/searchDistY) + np.abs(searchDistX/searchDistY - ref_score_a)
        #data0 = np.asarray([refDistX/searchDistX, refDistY/searchDistY])
        #data_arr.append(data0)
        #scoreC_arr.append(search_scoreC)
        #dist_x_arr.append(searchDistX)
        #dist_y_arr.append(searchDistY)
        #dist_z_arr.append(searchDistZ)
        data_arr.append(np.mean(search_tri_res, axis = 0))
        #ratio_zx_arr.append(searchDistZ/searchDistX)
        #ratio_zy_arr.append(searchDistZ/searchDistY)
        
        #scoreA_arr.append(search_scoreA)
        #scoreB_arr.append(search_scoreB)
        if(search_score < opt_search_score):
            opt_search_score = search_score
            opt_tmod = i
    print("Found T-mod: " + str(opt_tmod))
    
    #save arrays
    #dist_x_arr = np.asarray(dist_x_arr)
    #dist_y_arr = np.asarray(dist_y_arr)
    #dist_z_arr = np.asarray(dist_z_arr)
    #ratio_zx_arr = np.asarray(ratio_zx_arr)
    #ratio_zy_arr = np.asarray(ratio_zy_arr)
    #scoreA_arr = np.asarray(scoreA_arr)
    #scoreB_arr = np.asarray(scoreB_arr)
    #scoreC_arr = np.asarray(scoreC_arr)
    data_arr = np.asarray(data_arr)
    
    #np.savetxt("dist_x.txt",dist_x_arr)
    #np.savetxt("dist_y.txt",dist_y_arr)
    #np.savetxt("dist_z.txt",dist_z_arr)
    #np.savetxt("ratio_zx.txt", ratio_zx_arr)
    #np.savetxt("ratio_zy.txt", ratio_zy_arr)
    #np.savetxt("scoreA.txt", scoreA_arr)
    #np.savetxt("scoreB.txt", scoreB_arr)
    #np.savetxt("scoreC.txt", scoreC_arr)
    np.savetxt("means.txt",data_arr)
    '''
    opt_tmod = 0.38
    tri_res = scr.triangulate_list_nobar(ptsL,ptsR, r_vec, t_vec*opt_tmod, kL_inv, kR_inv, config.precise)
    minSearchX = np.min(tri_res[:,0])
    maxSearchX = np.max(tri_res[:,0])
    minSearchY = np.min(tri_res[:,1])
    maxSearchY = np.max(tri_res[:,1])
    minSearchZ = np.min(tri_res[:,2]) 
    maxSearchZ = np.max(tri_res[:,2])
    searchDistX = maxSearchX- minSearchX
    searchDistY = maxSearchY - minSearchY
    searchDistZ = maxSearchZ - minSearchZ
    print('Sch Dist X: ' + str(searchDistX))
    print('Sch Dist Y: ' + str(searchDistY))
    print('Sch Dist Z: ' + str(searchDistZ))
    print('_________')
    print('Sch Z/X: ' + str(searchDistZ/searchDistX))
    print('Sch Z/Y: ' + str(searchDistZ/searchDistY))
    print('_______')
    print('Ref X/Sch X: ' + str(refDistX/searchDistX))
    print('Ref Y/Sch Y: ' + str(refDistY/searchDistY))
    print('Ref Z/Sch Z: ' + str(refDistZ/searchDistZ))
    print('_________')
    '''
    


def t3_data0():
    folder_statue = "./test_data/statue/"
    input_data = "Rekonstruktion30.pcf"
    #Load arrays
    dist_x_arr = np.loadtxt("dist_x.txt")
    dist_y_arr = np.loadtxt("dist_y.txt")
    dist_z_arr = np.loadtxt("dist_z.txt")
    ratio_zx_arr = np.loadtxt("ratio_zx.txt")
    ratio_zy_arr = np.loadtxt("ratio_zy.txt")
    scoreA_arr = np.loadtxt("scoreA.txt")
    scoreB_arr = np.loadtxt("scoreB.txt") 
    scoreC_arr = np.loadtxt("scoreC.txt")
    scalefactor_arr = np.loadtxt("scale_factors.txt")
    means_arr = np.loadtxt("means.txt")
    #set X axis
    inc = 0.01
    max_tmod = 1.0
    x_rng = np.arange(inc,max_tmod,inc)
    #load reference data
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(folder_statue + input_data)
    geom_arr=remove_z_outlier(geom_arr, col_arr)
    minRefX = np.min(geom_arr[:,0])
    maxRefX = np.max(geom_arr[:,0])
    minRefY = np.min(geom_arr[:,1])
    maxRefY = np.max(geom_arr[:,1])
    minRefZ = np.min(geom_arr[:,2])
    maxRefZ = np.max(geom_arr[:,2])
    refDistX = maxRefX - minRefX
    refDistY = maxRefY - minRefY
    refDistZ = maxRefZ - minRefZ
    ref_zx = refDistZ/refDistX
    ref_zy = refDistZ/refDistY
    ref_xy = refDistX/refDistY
    ref_mean = np.mean(geom_arr,axis = 0)
    zx = np.abs(dist_z_arr/dist_x_arr - np.ones_like(x_rng)*ref_zx)
    zy = np.abs(dist_z_arr/dist_y_arr - np.ones_like(x_rng)*ref_zy)
    xy = np.abs(dist_x_arr/dist_y_arr - np.ones_like(x_rng)*ref_xy)
    means_zx = np.abs(means_arr[:,2]/means_arr[:,0] - np.ones_like(x_rng)*ref_mean[2]/ref_mean[0])
    means_zy = np.abs(means_arr[:,2]/means_arr[:,1] - np.ones_like(x_rng)*ref_mean[2]/ref_mean[1])
    means_xy = np.abs(means_arr[:,0]/means_arr[:,1] - np.ones_like(x_rng)*ref_mean[0]/ref_mean[1])
    
    plt.plot(x_rng, means_xy)
t3_data0()