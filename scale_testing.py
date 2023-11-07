# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:44:32 2023

@author: Admin
"""

import numpy as np
import scripts as scr
import numba
from tqdm import tqdm

def find_centroid(data):
    return np.asarray([np.mean(data[:,0]), np.mean(data[:,1]),np.mean(data[:,2])]) 
def accel_outliers():
    pass
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
    pass
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
    #Possibly run this on numba?
    res = []
    for i in tqdm(range(data.shape[0])):
        counter = 0
        counter = accel_dist_count(data, data[i,:], i, thresh_dist, thresh_count)
        if counter > thresh_count:
            res.append(data[i,:])
    return np.asarray(res)

def cleanup(data,thresh_dist_scale, thresh_count, outlier_scale):
    #Generate 
    pass
    
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
    print('')
    thresh_dist = dist_scale*2
    res_arr1 = remove_outliers(geom_arr, thresh_dist*80)
    print(' ')
    print(res_arr1.shape)
    res_arr2 = remove_noise(res_arr1, thresh_dist)
    scr.convert_np_ply(geom_arr,col_arr,'calib0.ply', overwrite = True)
    col_arr1 = scr.gen_color_arr_black(res_arr1.shape[0])
    scr.convert_np_ply(res_arr1,col_arr1,'calib1.ply', overwrite = True)
    col_arr2 = scr.gen_color_arr_black(res_arr2.shape[0])
    scr.convert_np_ply(res_arr2,col_arr2,'calib2.ply', overwrite = True)
    print(res_arr2.shape)
runA1()