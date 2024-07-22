# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:22:04 2024

@author: M
"""

import numpy as np
import scripts as scr
import numba
import os
from tqdm import tqdm
import cv2
float_epsilon = 1e-9
def create_sum_grid(img_stk):
    res_grid_stk = []

        
def startup_load(config):
    '''
    Loads inputs from config file. Also applies rectification and initial filters.    
    
    Parameters
    ----------
    config : confighandler object
        Configuration file loader

    Returns
    -------
    kL : 3x3 numpy array
        Left camera matrix
    kR : 3x3 numpy array
        Right camera matrix
    r_vec : 3x3 numpy array
        Rotation matrix
    t_vec : numpy array
        Translation vector
    kL_inv : 3x3 numpy array
        Inverse of left camera matrix
    kR_inv : 3x3 numpy array
        Inverse of right camera matrix
    fund_mat : 3x3 numpy array
        Fundamental matrix
    imgL : numpy array
        Input left camera images
    imgR : numpy array
        Input right camera images
    imshape : tuple
        shape of image inputs
    maskL : numpy array
        masked and filtered left images
    maskR : numpy array
        masked and filtered right images

    '''
    print("Loading files...")
    kL,kR,r_vec,t_vec = scr.load_mats(config.mat_folder, config.kL_file, 
                                         config.kR_file, config.R_file, config.t_file, 
                                         config.skiprow,config.delim)
    #Load images
    if(config.sing_img_mode):
        imgL,imgR = scr.load_images_1_dir(config.sing_img_folder, config.sing_left_ind, config.sing_right_ind, config.sing_ext)
    else:
        imgL,imgR = scr.load_images(folderL = config.left_folder, folderR = config.right_folder)
    imshape = imgL[0].shape
    
    #undistort images if set for
    if config.distort_comp:
        #load distortion vectors
        dL = np.loadtxt(config.mat_folder +config.left_distort, skiprows=config.skiprow, delimiter = config.delim)
        dR = np.loadtxt(config.mat_folder +config.right_distort, skiprows=config.skiprow, delimiter = config.delim)
        #undistort images and update camera matrices
        kL,imgL= scr.undistort(imgL, kL,dL)
        kR,imgR= scr.undistort(imgR, kR,dR)
   
    #rectify images
    fund_mat = None
    if os.path.isfile(config.mat_folder + config.f_file) and config.f_mat_file_mode == 1:
        fund_mat = np.loadtxt(config.mat_folder + config.f_file, skiprows=config.skiprow, delimiter = config.delim)
        print("Fundamental Matrix Loaded From File: " + config.mat_folder + config.f_file)
    else:
        F=None
        if config.f_search:
            F = scr.find_f_mat_list(imgL,imgR, config.f_mat_thresh, config.f_calc_mode)
        else:
            if(config.f_mat_ncc):
                F = scr.find_f_mat_ncc(imgL,imgR,config.f_mat_thresh, config.f_calc_mode)
            else:
                
                F = scr.find_f_mat(imgL[0],imgR[0], config.f_mat_thresh, config.f_calc_mode)
        if config.f_mat_file_mode == 2:
            print("Fundamental Matrix Saved To File: " + config.mat_folder + config.f_file)
            np.savetxt(config.mat_folder + config.f_file, F)
            with open(config.mat_folder + config.f_file, 'r') as ori:
                oricon = ori.read()
            with open(config.mat_folder + config.f_file, 'w') as ori:  
                ori.write("3\n3\n")
                ori.write(oricon)
        fund_mat = F
    rectL,rectR = scr.rectify_lists(imgL,imgR, fund_mat)
    avgL = np.asarray(rectL).mean(axis=(0))
    avgR = np.asarray(rectR).mean(axis=(0))

    #Background filter
    thresh_val = config.mask_thresh
    maskL = scr.mask_avg_list(avgL,rectL, thresh_val)
    maskR = scr.mask_avg_list(avgR,rectR, thresh_val)

    maskL = np.asarray(maskL)
    maskR = np.asarray(maskR)
    col_refL = None
    col_refR = None
    if config.color_recon:
        col_refL, col_refR= scr.load_images_1_dir(config.sing_img_folder, config.sing_left_ind, config.sing_right_ind, config.sing_ext, colorIm = True)
    
    return kL, kR, r_vec, t_vec, fund_mat, imgL, imgR, imshape, maskL, maskR, col_refL, col_refR


