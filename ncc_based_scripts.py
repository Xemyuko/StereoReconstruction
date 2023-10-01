# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:56:07 2023

@author: Admin
"""
import numpy as np
import scripts as scr
import ncc_core as ncc
from tqdm import tqdm
def ref_tmod_find(config):
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(config.ref_data_file)
    geom_arr = scr.remove_z_outlier_no_col(geom_arr)
    minRefX = np.min(geom_arr[:,0])
    maxRefX = np.max(geom_arr[:,0])
    minRefY = np.min(geom_arr[:,1])
    maxRefY = np.max(geom_arr[:,1])
    minRefZ = np.min(geom_arr[:,2])
    maxRefZ = np.max(geom_arr[:,2])
    refDistX = maxRefX - minRefX
    refDistY = maxRefY - minRefY
    refDistZ = maxRefZ - minRefZ
    inc = 0.01
    max_tmod = config.max_tmod
    opt_search_score = 100
    opt_tmod = inc
    ref_zx = refDistZ/refDistX
    ref_zy = refDistZ/refDistY
    ref_xy = refDistX/refDistY
    kL, kR, r_vec, t_vec = scr.initial_load(1.0,config.mat_folder)
    kL_inv = np.linalg.inv(kL)
    kR_inv = np.linalg.inv(kR)
    ptsL,ptsR = ncc.cor_internal(config)
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
        ptsL,ptsR = ncc.cor_internal(config)
        search_score = (np.abs(searchDistZ/searchDistX - ref_zx) + np.abs(searchDistZ/searchDistY - ref_zy) + np.abs(searchDistX/searchDistY - ref_xy))/3 
        if(search_score < opt_search_score):
            opt_search_score = search_score
            opt_tmod = i
    return opt_tmod