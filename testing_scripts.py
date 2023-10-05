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
import cv2
import json

def t1():
    folder = "./test_data/calibObjects/"
    filename = folder + 'testcuberead.json'
    f = open(filename)
    data = json.load(f)
    print(data['objects'][0]['vertices'])

    f.close()
t1()    
def t2():
    #set reference pcf file, folders for images and matrices. 
    folder_statue = "./test_data/statue/"
    matrix_folder = "matrix_folder/"
    left_folder = "camera_L/"
    right_folder = "camera_R/"
    input_data = "Rekonstruktion30.pcf"
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(folder_statue + input_data)
    kL, kR, r_vec, t_vec = scr.initial_load(1.0,folder_statue + matrix_folder)
    projR = np.append(r_vec,t_vec,axis = 1) 
    projL = np.append(np.identity(3),np.asarray([[0],[0],[0]]),axis = 1)
    P1 = kL @ projL 
    P2 = kR @ projR
    homogeneous_3D_points = cv2.triangulatePoints(P1, P2, xy1.T, xy2.T)
    points_res = (homogeneous_3D_points / homogeneous_3D_points[3])[:3].T
    scr.convert_np_ply(points_res, scr.gen_color_arr_black(points_res.shape[0]), 't2recon')
    

def distance3D(pt1,pt2):
    res = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)
    return res



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
    geom_arr, col_arr=scr.remove_z_outlier(geom_arr, col_arr)
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
        
        search_score = np.abs(searchDistZ/searchDistX - ref_score_x) + np.abs(searchDistZ/searchDistY - ref_score_y) 

       # data_arr.append(np.mean(search_tri_res, axis = 0))

        

        if(search_score < opt_search_score):
            opt_search_score = search_score
            opt_tmod = i
    print("Found T-mod: " + str(opt_tmod))
    
    #save arrays

    #data_arr = np.asarray(data_arr)
    

    #np.savetxt("means.txt",data_arr)
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
    
def t3_plot():
    folder_statue = "./test_data/statue/"
    input_data = "Rekonstruktion30.pcf"
    #Load arrays
    dist_x_arr = np.loadtxt("dist_x.txt")
    dist_y_arr = np.loadtxt("dist_y.txt")
    dist_z_arr = np.loadtxt("dist_z.txt")
    scalefactor_arr = np.loadtxt("scale_factors.txt")
    means_arr = np.loadtxt("means.txt")
    #set X axis
    inc = 0.01
    max_tmod = 1.0
    x_rng = np.arange(inc,max_tmod,inc)
    #load reference data
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(folder_statue + input_data)
    geom_arr, col_arr=scr.remove_z_outlier(geom_arr, col_arr)
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
    #Comparison of zx and zy ratios with reference
    scoreA_arr_calc = zx + zy
    plt.title('X Dist')
    plt.plot(x_rng,dist_x_arr)
    plt.show()
    plt.title('Y Dist')
    plt.plot(x_rng,dist_y_arr)
    plt.show()
    plt.title('Z Dist')
    plt.plot(x_rng,dist_z_arr)
    plt.show()
    plt.title('XYZ Dist')
    plt.plot(x_rng,dist_x_arr, label = 'X')
    plt.plot(x_rng,dist_y_arr, label = 'Y')
    plt.plot(x_rng,dist_z_arr, label = 'Z')
    plt.legend()
    plt.show()
    plt.title('Z/X')
    plt.plot(x_rng, dist_z_arr/dist_x_arr)
    plt.show()
    plt.title('Z/Y')
    plt.plot(x_rng, dist_z_arr/dist_y_arr)
    plt.show()
    plt.title('Scale factors')
    plt.plot(x_rng, (np.ones_like(x_rng) * refDistX)/dist_x_arr, label = 'X')
    plt.plot(x_rng, (np.ones_like(x_rng) * refDistY)/dist_y_arr, label = 'Y')
    plt.plot(x_rng, (np.ones_like(x_rng) * refDistZ)/dist_z_arr, label = 'Z')
    plt.legend()
    plt.show()
    plt.title('Search Score zx+zy')
    plt.plot(x_rng,scoreA_arr_calc)
    plt.show()
    print('Min Score at: t_mod = ' + str(x_rng[np.argmin(scoreA_arr_calc)]))
    plt.title('Search Score avg(zy,zx,xy)')
    plt.plot(x_rng,(zx+zy+xy)/3)
    plt.show()
    print('Min Score at: t_mod = ' + str(x_rng[np.argmin(scoreA_arr_calc)]))
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
    geom_arr, col_arr=scr.remove_z_outlier(geom_arr, col_arr)
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
