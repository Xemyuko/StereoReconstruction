# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:38:01 2023

@author: Admin
"""

import scripts as scr
import confighandler as chand
import ncc_core as ncc
import numpy as np
config = chand.ConfigHandler()
config.load_config()
kL,kR,r_vec,t_vec = scr.initial_load(config.tmod, config.mat_folder, config.kL_file, 
                                     config.kR_file, config.R_file, config.t_file, 
                                     config.skiprow,config.delim)

def computeF(kL,kR,r_vec,t_vec):
    A = kL @ r_vec.T @ t_vec
    C = [[0, -A[2][0], A[1][0]] ,[A[2][0] ,0, -A[0][0]], [-A[1][0], A[0][0], 0]]
    return (np.linalg.inv(kR)).T @ r_vec @ kL.T @ C
imgfold1 = "./test_data/statue/camera_L/"
imgfold2 = "./test_data/statue/camera_R/"
imgs1 = scr.load_imgs_1_dir(imgfold1)
imgs2 = scr.load_imgs_1_dir(imgfold2)
stack_loc = 20
stack1,stack2 = scr.get_pix_stack(imgs1,imgs2,stack_loc,stack_loc) 
scr.display_stereo(stack1,stack2)
F = scr.find_f_mat(imgs1[0],imgs2[0])
rectL,rectR = scr.rectify_lists(imgs1,imgs2, F)
avgL = np.asarray(rectL).mean(axis=(0))
avgR = np.asarray(rectR).mean(axis=(0))
stack3,stack4 = scr.get_pix_stack(rectL,rectR,stack_loc,stack_loc) 
scr.display_stereo(stack3,stack4)