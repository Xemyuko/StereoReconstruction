# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:38:01 2023

@author: Admin
"""

import scripts as scr
import confighandler as chand
import ncc_core as ncc
import numpy as np
version = 0
config = chand.ConfigHandler(version)
config.load_config()
kL,kR,r_vec,t_vec = scr.initial_load(config.tmod, config.mat_folder, config.kL_file, 
                                     config.kR_file, config.R_file, config.t_file, 
                                     config.skiprow,config.delim)

def computeF(kL,kR,r_vec,t_vec):
    A = kL @ r_vec.T @ t_vec
    C = [[0, -A[2][0], A[1][0]] ,[A[2][0] ,0, -A[0][0]], [-A[1][0], A[0][0], 0]]
    return (np.linalg.inv(kR)).T @ r_vec @ kL.T @ C

F = computeF(kL,kR,r_vec,t_vec)
print(F)