# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:22:58 2023

@author: myuey
"""
import scripts as scr
import numpy as np
import ncc_core as ncc
import confighandler as ch


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
config.left_folder = left_folder
config.right_folder = right_folder
config.precise = 1
config.corr_map_out = 0
#use internal reconstruction
tri_res, cor_map = ncc.run_cor_internal(config)