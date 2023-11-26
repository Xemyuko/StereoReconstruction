# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:44:32 2023

@author: Admin
"""

import numpy as np
import scripts as scr
import numba
from tqdm import tqdm
import json

def find_centroid(data):
    return np.asarray([np.mean(data[:,0]), np.mean(data[:,1]),np.mean(data[:,2])]) 

def remove_outliers(data, thresh_dist):
    p1 = find_centroid(data)
    res_arr = []
    for i in range(data.shape[0]):
        p2 = data[i,:]
        dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
        if dist < thresh_dist:
            res_arr.append(p2)
    return np.asarray(res_arr)
def sort_depths(data):
    #Sort by depth value z
    data = data[data[:, 2].argsort()]
    return data
@numba.jit(nopython=True)
def accel_dist_count(data, data_point, data_ind, thresh_dist, thresh_count):
    counter = 0
    for i in range(data.shape[0]):
        if i != data_ind:
            p1 = data[i]
            p2=data_point
            dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
            if(dist < thresh_dist):
                counter+=1
                if counter > thresh_count:
                    break
    return counter
def remove_noise(data,thresh_dist, thresh_count = 60):
    #Calculate the distance from a chosen point to all other points. Stop if counter of points
    #within threshold distance reaches check threshold. If stopped, add point to new data array. 
    res = []
    for i in tqdm(range(data.shape[0])):
        counter = 0
        counter = accel_dist_count(data, data[i,:], i, thresh_dist, thresh_count)
        if counter > thresh_count:
            res.append(data[i,:])
    return np.asarray(res)

def cleanup(data,dist_val,noise_scale, thresh_count, outlier_scale, n_rand = 5):
    
    #Run code for removing outliers on data and distance threshold
    centroid = np.asarray([np.mean(data[:,0]), np.mean(data[:,1]),np.mean(data[:,2])]) 
    res_arr_out = []
    for i in range(data.shape[0]):
        out2 = data[i,:]
        dist = np.sqrt((centroid[0]-out2[0])**2 + (centroid[1]-out2[1])**2 + (centroid[2]-out2[2])**2)
        if dist < outlier_scale * dist_val:
            res_arr_out.append(out2)
    res_arr_out = np.asarray(res_arr_out)
    #Run code for removing noise on result of above
    res = []
    for i in range(res_arr_out.shape[0]):
        counter = 0
        counter = accel_dist_count(res_arr_out, res_arr_out[i,:], i, noise_scale * dist_val, thresh_count)
        if counter > thresh_count:
            res.append(res_arr_out[i,:])
    #Return cleaned data
    return np.asarray(res)
def convert_cad_ply(filename):
    f = open(filename)
    data = json.load(f)
    
    f.close()
    res = data['objects'][0]['vertices']
    col = scr.gen_color_arr_black(len(res))
    scr.convert_np_ply(res,col,'vertices_test.ply')

def identify_planes(data, dist_scale, align = False, plane_length_count = 100):
    #Sort data by depth
    data = data[data[:, 2].argsort()]
    #Split sorted data in half
    dataA = data[0:int(data.shape[0]/2),:]
    dataB = data[int(data.shape[0]/2):data.shape[0],:]
    #Find centroids of each half
    centroidA = np.asarray([np.mean(dataA[:,0]), np.mean(dataA[:,1]),np.mean(dataA[:,2])]) 
    centroidB = np.asarray([np.mean(dataB[:,0]), np.mean(dataB[:,1]),np.mean(dataB[:,2])]) 
    #Split data by X value around each centroid
    dataA_1 = []
    dataA_2 = []
    for i in dataA:
        val_check = i[0]
        if(val_check < centroidA[0]):
            dataA_1.append(i)
        else:
            dataA_2.append(i)
    dataA_1 = np.asarray(dataA_1)
    dataA_2 = np.asarray(dataA_2)
    dataB_1 = []
    dataB_2 = []
    for i in dataB:
        val_check = i[0]
        if(val_check < centroidB[0]):
            dataB_1.append(i)
        else:
            dataB_2.append(i)
    dataB_1 = np.asarray(dataB_1)
    dataB_2 = np.asarray(dataB_2)
    #Find centroids of the splits
    centroidA_1 = np.asarray([np.mean(dataA_1[:,0]), np.mean(dataA_1[:,1]),np.mean(dataA_1[:,2])])
    centroidA_2 = np.asarray([np.mean(dataA_2[:,0]), np.mean(dataA_2[:,1]),np.mean(dataA_2[:,2])])
    centroidB_1 = np.asarray([np.mean(dataA_1[:,0]), np.mean(dataA_1[:,1]),np.mean(dataA_1[:,2])])
    centroidB_2 = np.asarray([np.mean(dataA_2[:,0]), np.mean(dataA_2[:,1]),np.mean(dataA_2[:,2])])
    #Align centroids on the same X as the main centroid of each region
    alignCenA_1 = np.asarray([centroidA[0], centroidA_1[1], centroidA_1[2]])
    alignCenA_2 = np.asarray([centroidA[0], centroidA_2[1], centroidA_2[2]])
    alignCenB_1 = np.asarray([centroidB[0], centroidB_1[1], centroidB_1[2]])
    alignCenB_2 = np.asarray([centroidB[0], centroidB_2[1], centroidB_2[2]])
    #Create planes with centroids
    if(align):
        plane_tripletA = np.asarray([alignCenA_1, centroidA, alignCenA_2])
        print(plane_tripletA)
        plane_tripletB = np.asarray([alignCenB_1, centroidB, alignCenB_2])
        planeA = cre_pl_pts(dist_scale, plane_tripletA, plane_length_count)
        planeB = cre_pl_pts(dist_scale, plane_tripletB, plane_length_count)
        return planeA,planeB
    else:
        plane_tripletA = np.asarray([centroidA_1, centroidA, centroidA_2])
        plane_tripletB = np.asarray([centroidB_1, centroidB, centroidB_2])
        planeA = cre_pl_pts(dist_scale, plane_tripletA, plane_length_count)
        planeB = cre_pl_pts(dist_scale, plane_tripletB, plane_length_count)
        return planeA,planeB
def id_plane2(data,dist_scale):
    #calculate centroid of data set
    centroid = np.asarray([np.mean(data[:,0]), np.mean(data[:,1]),np.mean(data[:,2])]) 
    #Split data around the centroid by depth
    #find centroids of each half
def cre_pl_pts(dist_scale, plane_triplet, plane_length_count):
    p0, p1, p2 = plane_triplet
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    ux, uy, uz = [x1-x0, y1-y0, z1-z0]
    vx, vy, vz = [x2-x0, y2-y0, z2-z0]

    u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

    point  = np.array(p0)
    normal = np.array(u_cross_v)

    d = -point.dot(normal)
    res_pts = []
    for i in range(-int(plane_length_count/2),int(plane_length_count/2)):
        for j in range(-int(plane_length_count/2),int(plane_length_count/2)):
            xx = i*dist_scale + plane_triplet[0][0] 
            yy = j*dist_scale + plane_triplet[0][1] 
            z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
            pt_entry = [xx,yy,z]
            res_pts.append(pt_entry)
    return np.asarray(res_pts)

def runA2():
    filename = './test_data/calibObjects/panel_calib.json'
    convert_cad_ply(filename)  
def runA3():
    data_filepath = './test_data/calibObjects/000POS0Rekonstruktion30.pcf'   
    xy1,xy2,data,col_arr,correl = scr.read_pcf(data_filepath)     
    #Sort data by depth
    data = data[data[:, 2].argsort()]
    #Split sorted data in half
    dataA = data[0:int(data.shape[0]/2),:]
    dataB = data[int(data.shape[0]/2):data.shape[0],:]
    #Find centroids of each half
    centroidA = np.asarray([np.mean(dataA[:,0]), np.mean(dataA[:,1]),np.mean(dataA[:,2])]) 
    centroidB = np.asarray([np.mean(dataB[:,0]), np.mean(dataB[:,1]),np.mean(dataB[:,2])]) 
def runA1(): 
    data_filepath = './test_data/calibObjects/000POS0Rekonstruktion30.pcf'
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(data_filepath)
    '''
    minRefX = np.min(geom_arr[:,0])
    maxRefX = np.max(geom_arr[:,0])
    minRefY = np.min(geom_arr[:,1])
    maxRefY = np.max(geom_arr[:,1])
    minRefZ = np.min(geom_arr[:,2])
    maxRefZ = np.max(geom_arr[:,2])
    refDistX = maxRefX - minRefX
    refDistY = maxRefY - minRefY
    refDistZ = maxRefZ - minRefZ
    '''
    print(geom_arr.shape)
    p1 = geom_arr[10]
    p2 = geom_arr[11]
    dist_scale = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    print('SCALE CHECK: ' + str(dist_scale))
    print('CLEANUP')
    res_arr3 = cleanup(geom_arr, dist_scale, 2, 60, 160)
    print(res_arr3.shape)
    col_arr3 = scr.gen_color_arr_black(res_arr3.shape[0])
    scr.convert_np_ply(res_arr3,col_arr3,'calib2.ply', overwrite = True)
    
    #pick 3 points from list
    plane_triplet = [res_arr3[44], res_arr3[59], res_arr3[78]]
    res_pts = cre_pl_pts(dist_scale, plane_triplet, 100)
    res_arr4 = np.concatenate((res_arr3,res_pts))
    col_arr4 = scr.gen_color_arr_black(res_arr4.shape[0])
    scr.convert_np_ply(res_arr4,col_arr4,'calib_plane.ply', overwrite = True)
    
    planeA1, planeB1 = identify_planes(res_arr3, dist_scale)
    #planeA2, planeB2 = identify_planes(res_arr3, dist_scale, align = True)
    planesA = np.concatenate((planeA1,planeB1))
    #planesB = np.concatenate((planeA2,planeB2))
    
    planes_no_align = np.concatenate((res_arr3,planesA))
   # planes_align = np.concatenate((res_arr3,planesB))
    col_no_align = scr.gen_color_arr_black(planes_no_align.shape[0])
    #col_align = scr.gen_color_arr_black(planes_align.shape[0])
    scr.convert_np_ply(planes_no_align,col_no_align, "plane_no_align.ply", overwrite = True)
   # scr.convert_np_ply(planes_align,col_align, "plane_align.ply", overwrite = True)
    
runA1()