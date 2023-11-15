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

def identify_planes(data, dist_scale):
   #Sort data by depth
   data = data[data[:, 2].argsort()]
   #Bisect sorted data in half
   dataA = data[0:int(data.shape[0]/2),:]
   dataB = data[int(data.shape[0]/2):data.shape[0],:]
   col_arrA = scr.gen_color_arr_black(dataA.shape[0])
   col_arrB = scr.gen_color_arr_black(dataB.shape[0])
   scr.convert_np_ply(dataA,col_arrA,'calibA.ply', overwrite = True)
   scr.convert_np_ply(dataB,col_arrB,'calibB.ply', overwrite = True) 
   dataC = cleanup(dataA, dist_scale, 2, 40, 50)
   scr.convert_np_ply(dataC,col_arrA,'calibC.ply', overwrite = True)
def create_plane_circles(dist_scale, plane_triplet, circle_num, circle_scale):
    #Generate plane equation 
    #Determine vector AB and BC
    vec_ab = plane_triplet[0]-plane_triplet[1]
    vec_bc = plane_triplet[1]-plane_triplet[2]
    #cross-product
    norm_vec = np.cross(vec_ab,vec_bc)
    d = np.sum(norm_vec * plane_triplet[0])
    #plane equation => norm_vec * point + d = 0
    #Calculate center point of plane_triplet
    plane_triplet = np.asarray(plane_triplet)
    center = np.asarray([np.mean(plane_triplet[0,:]), np.mean(plane_triplet[1,:]), np.mean(plane_triplet[2,:])])
    #Convert to spherical coordinates with the center point as the origin to determine the angle for the plane
    #by the planr triplet
    #Find plane triplet points in terms of position from center, with center as (0,0,0)
    centered_plane_triplet =[]
    centered_plane_triplet.append(plane_triplet[0] - center)
    centered_plane_triplet.append(plane_triplet[1] - center)
    centered_plane_triplet.append(plane_triplet[2] - center)
def create_plane_pts(dist_scale, plane_triplet, plane_length_count):
    res = []
    #Generate plane equation 
    #Determine vector AB and BC
    vec_ab = plane_triplet[0]-plane_triplet[1]
    vec_bc = plane_triplet[1]-plane_triplet[2]
    #cross-product
    norm_vec = np.cross(vec_ab,vec_bc)
    d = np.sum(norm_vec * plane_triplet[0])
    #plane equation => norm_vec * point + d = 0
    
    #Center plane around first point in plane_triplet
    for i in range(-int(plane_length_count/2),int(plane_length_count/2)):
        for j in range(-int(plane_length_count/2),int(plane_length_count/2)):
            x_test = i*dist_scale + plane_triplet[0][0] 
            y_test = j*dist_scale + plane_triplet[0][1] 
            x_test*norm_vec[0] + y_test*norm_vec[1]
def runA2():
    filename = './test_data/calibObjects/panel_calib.json'
    convert_cad_ply(filename)            
def runA1(): 
    data_filepath = './test_data/calibObjects/000POS0Rekonstruktion30.pcf'
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(data_filepath)
    minRefX = np.min(geom_arr[:,0])
    maxRefX = np.max(geom_arr[:,0])
    minRefY = np.min(geom_arr[:,1])
    maxRefY = np.max(geom_arr[:,1])
    minRefZ = np.min(geom_arr[:,2])
    maxRefZ = np.max(geom_arr[:,2])
    refDistX = maxRefX - minRefX
    refDistY = maxRefY - minRefY
    refDistZ = maxRefZ - minRefZ
    print(geom_arr.shape)
    p1 = geom_arr[10]
    p2 = geom_arr[11]
    dist_scale = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    print('SCALE CHECK: ' + str(dist_scale))
    print('CLEANUP')
    res_arr3 = cleanup(geom_arr, dist_scale, 2, 60, 160)
    print(res_arr3.shape)
   # col_arr3 = scr.gen_color_arr_black(res_arr3.shape[0])
   # scr.convert_np_ply(res_arr3,col_arr3,'calib2.ply', overwrite = True)
    identify_planes(res_arr3, dist_scale)
runA2()