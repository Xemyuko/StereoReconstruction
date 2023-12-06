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

def test_fix2():
    #Load Matrices
    testFolder = "./test_data/maustest/"
    input_data = "Maus1.pcf"
    skiprow = 2
    delim = ' '
    kL = np.loadtxt(testFolder + 'Kl.txt', skiprows=skiprow, delimiter = delim)
    kR = np.loadtxt(testFolder +  'Kr.txt', skiprows=skiprow, delimiter = delim)
    kL_inv = np.linalg.inv(kL)
    kR_inv = np.linalg.inv(kR)
    r_vec = np.loadtxt(testFolder +  'R.txt', skiprows=skiprow, delimiter = delim)
    t_vec = np.loadtxt(testFolder +  't.txt', skiprows=skiprow, delimiter = delim)
    #Access 2D points from reference pcf
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(testFolder + input_data)
    res=[]
    for i,j in zip(xy1,xy2):
        res.append(scr.triangulate_avg(i,j,r_vec,t_vec, kL_inv, kR_inv ))
    res = np.asarray(res)
    scr.create_ply(res, "testmaus2")
test_fix2()
def test_fix():
    #Load Matrices
    testFolder = "./test_data/maustest/"
    input_data = "Maus1.pcf"
    skiprow = 2
    delim = ' '
    kL = np.loadtxt(testFolder + 'Kl.txt', skiprows=skiprow, delimiter = delim)
    kR = np.loadtxt(testFolder +  'Kr.txt', skiprows=skiprow, delimiter = delim)
    r_vec = np.loadtxt(testFolder +  'R.txt', skiprows=skiprow, delimiter = delim)
    t_vec = np.loadtxt(testFolder +  't.txt', skiprows=skiprow, delimiter = delim)
    #Create calc matrices 

    k1 = np.c_[kL, np.asarray([[0],[0],[1]])]

    k2 = np.c_[kR, np.asarray([[0],[0],[1]])]
    RT = np.c_[r_vec, t_vec]
    RT = np.r_[RT, [np.asarray([0,0,0,1])]]
    
    P1 = k1 @ np.eye(4,4)
    print(P1)
    print(P1[2,:])
    P2 = k2 @ RT

    #Access 2D points from reference pcf
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(testFolder + input_data)
    res=[]
    cor_thresh = 0.9
    ignore_thresh = True
    for i,j,k in zip(xy1,xy2, correl):
        #Check correlation threshold
        
        if(k >= cor_thresh or ignore_thresh):
            #Create solution matrix
            sol0 = i[0] * P1[2,:] - P1[0,:]
            sol1 = i[1] * P1[2,:] - P1[1,:]
            sol2 = j[0] * P2[2,:] - P2[0,:]
            sol3 = j[1] * P2[2,:] - P2[1,:]
        
            solMat = np.stack((sol0,sol1,sol2,sol3))
            #Apply SVD to solution matrix to find triangulation
            U,s,vh = np.linalg.svd(solMat,full_matrices = True)
            Q = vh[:,3]

            Q *= 1/Q[3]
            res.append(Q[:3])
    res = np.asarray(res)
    filt_res = []
    filt_thresh = 0.9
    for i,j in zip(geom_arr,correl):
        if(j >= filt_thresh):
            filt_res.append(i)
    filt_res = np.asarray(filt_res)
    scr.create_ply(res, "testmaus")
    scr.create_ply(geom_arr, "referencemaus")
    scr.create_ply(filt_res, 'filtmaus')

def t1():
    folder = "./test_data/calibObjects/"
    filename = folder + 'testconeread.json'
    f = open(filename)
    data = json.load(f)
    vertices = np.asarray(data['objects'][0]['vertices'])
    f.close()
    maxVals = np.max(vertices, axis = 0)
    minVals = np.min(vertices, axis = 0)
    refXdist = maxVals[0] - minVals[0]
    refYdist = maxVals[2] - minVals[2]
    refZdist = maxVals[1] - minVals[1]
    chkZdist = 40
    print(data['objects'][0])
  
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
def cluster_dim():
    pass
def clustertest0():
    #set reference pcf file, folders for images and matrices. 
    folder_statue = "./test_data/statue/"
    matrix_folder = "matrix_folder/"
    left_folder = "camera_L/"
    right_folder = "camera_R/"
    input_data = "Rekonstruktion30.pcf"
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
    #Run triangulation on ptsL and ptsR
    kL, kR, r_vec, t_vec = scr.initial_load(1.0,folder_statue + matrix_folder)
    kL_inv = np.linalg.inv(kL)
    kR_inv = np.linalg.inv(kR)
    #Run triangulation with tmod = 0.5 for scale factors
    scale_tri_res = scr.triangulate_list_nobar(ptsL,ptsR, r_vec, t_vec*0.5, kL_inv, kR_inv, config.precise)
    scale_tri_res = np.asarray(scale_tri_res)
    minScaleX = np.min(scale_tri_res[:,0])
    maxScaleX = np.max(scale_tri_res[:,0])
    minScaleY = np.min(scale_tri_res[:,1])
    maxScaleY = np.max(scale_tri_res[:,1])
    minScaleZ = np.min(scale_tri_res[:,2]) 
    maxScaleZ = np.max(scale_tri_res[:,2]) 
    scaleDistX = maxScaleX - minScaleX
    scaleDistY = maxScaleY - minScaleY
    scaleDistZ = maxScaleZ - minScaleZ
    factorX = scaleDistX/refDistX
    factorY = scaleDistY/refDistY
    factorZ = scaleDistZ/refDistZ
    #Generate cluster centers in reference 
    num_clusters = 3
    cluster_list = []
    for i in range(num_clusters):
        randGen = np.random.rand(3)
        cluster_list.append([randGen[0]*refDistX + minRefX, randGen[1]*refDistY + minRefY, randGen[2]*refDistZ + minRefZ])
    #scale cluster centers to tmod testing space dimensions
    cluster_chk_list = []
    for i in cluster_list:
        cluster_chk_list.append([i[0] * factorX, i[1] * factorY, i[2] * factorZ])
    #match number of data points and Reduce data point totals to manageable levels, n >= 10
    
    num_pts_skip = 100
    ratio_skip = np.round(scale_tri_res.shape[0]/geom_arr.shape[0], decimals = 2)
    testPtsL = []
    testPtsR = []
    refGeom = []
    for i in np.arange(0,geom_arr.shape[0],num_pts_skip):
        refGeom.append(geom_arr[i,:])
    for i in np.arange(0, len(ptsL), int(num_pts_skip*ratio_skip)):
        testPtsL.append(ptsL[i])
        testPtsR.append(ptsR[i])
    
    dist_arr_ref = []
    #Calculate 3D distances to each cluster center for each point
    for i in refGeom:
        dist_entry = [distance3D(cluster_list[0], i), distance3D(cluster_list[1], i),distance3D(cluster_list[2], i)]
        dist_arr_ref.append(dist_entry)
    #assign reference points to clusters 
    cluster_assign_ref = []
    for i in dist_arr_ref:
        cluster_assign_ref.append(np.argmax(i))
    unique, counts = np.unique(cluster_assign_ref, return_counts=True)    
    ref_dict = dict(zip(unique, counts))
    counts_ref = np.asarray(counts)
    print(ref_dict)
    print(len(dist_arr_ref))
    print(len(testPtsL))
    #test clustering scales
    test_tri_res = scr.triangulate_list_nobar(testPtsL,testPtsR, r_vec, t_vec*0.38, kL_inv, kR_inv, config.precise)
    dist_arr_test = []
    for a in test_tri_res:
        dist_entry = [distance3D(cluster_chk_list[0], a), distance3D(cluster_chk_list[1], a),distance3D(cluster_chk_list[2], a)]
        dist_arr_test.append(dist_entry)
    cluster_assign_test = []
    for b in dist_arr_test:
        cluster_assign_test.append(np.argmax(b))
    unique_test, counts_test = np.unique(cluster_assign_test, return_counts=True) 
    test_dict = dict(zip(unique_test, counts_test))
    print(test_dict)
    '''
    #repeat above for each iteration of tmod
    #search score is composite of absolute differences in xyz dimension ratios for each cluster 
    #and/or absolute differences in point counts of each cluster
    #loop through adjusting triangulation by varying the tmod scale factor and comparing the scaled result with the 
    inc = 0.01
    max_tmod = 1.0
    opt_search_score = 10000000
    opt_tmod = inc
    data_arr = []
    opt_clustering = []
    for i in tqdm(np.arange(inc,max_tmod,inc)):
        search_tri_res = scr.triangulate_list_nobar(ptsL,ptsR, r_vec, t_vec*i, kL_inv, kR_inv, config.precise)
        dist_arr_chk = []
        for a in search_tri_res:
            dist_entry = [distance3D(cluster_chk_list[0], a), distance3D(cluster_chk_list[1], a),distance3D(cluster_chk_list[2], a)]
            dist_arr_chk.append(dist_entry)
        cluster_assign_chk = []
        for b in dist_arr_chk:
            cluster_assign_chk.append(np.argmax(b))
        unique_chk, counts_chk = np.unique(cluster_assign_chk, return_counts=True)  
        if(len(unique_chk) == len(unique)):
            diff_chk = np.abs(np.asarray(counts_chk) - counts_ref)
            search_score = np.sum(diff_chk)
            data_arr.append(search_score)
            if(search_score < opt_search_score):
                opt_tmod = i
                opt_search_score = search_score
                opt_clustering = counts_chk
    print("Found T-mod: " + str(opt_tmod))
    print(opt_clustering)
    data_arr = np.asarray(data_arr)
    

    np.savetxt("cluster_check.txt",data_arr)
    '''

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
def create_calib_ply():
    input_data_path = "./test_data/calibObjects/000POS0Rekonstruktion30.pcf"
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(input_data_path)
    scr.convert_np_ply(geom_arr, col_arr, "test_calib_object.ply")
def find_first_surface(geom_arr):
    #Attempts to locate the points that make up the first perpendicular to camera 
    #view surface in a point cloud
    #Identify centroid 
    #identify min depth point and max depth point
    #set point quantity thresholds for surface slice
    #separate into buckets to section off 
    pass
def identify_surface(geom_arr):
    n = geom_arr.shape[0]
    yz1 = np.concatenate((geom_arr[:, 1:], np.ones((n, 1))), axis=1)

    p, *_ = np.linalg.lstsq(a=yz1, b=-geom_arr[:, 0], rcond=None)

    print('Desired vs. fit x:')
    print(np.stack((geom_arr[:, 0], -yz1 @ p)).T)
def tee3():
    input_data_path = "./test_data/calibObjects/000POS0Rekonstruktion30.pcf"
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(input_data_path)
    res = identify_surface(geom_arr)