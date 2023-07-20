# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:38:55 2023

@author: Admin
"""

import scripts as scr

def calibrate_save():
    left_cal_folder = "./test_data/schachbrett 230718/c1/"
    right_cal_folder = "./test_data/schachbrett 230718/c2/"
    calmtx_folder = "./cal_mtx/"
    rows = 8
    columns = 12
    world_scaling = 0.04
    #run camera calibration code on images
    kL, kR, distL, distR, R, T, E, F = scr.calibrate_cameras(left_cal_folder, right_cal_folder, "", 
                                                               rows, columns, world_scaling)
    scr.fill_mtx_dir(calmtx_folder, kL, kR, F, E, distL, distR, R, T)
#load sphere images
sphere_leftA = ""
sphere_rightA = ""

sphere_leftB = ""
sphere_rightB = ""

left_imgs, right_imgs = scr.load_images(sphere_leftA,sphere_rightA)
#load sphere matrices
tmod = 1

#run test reconstruction of sphere with tmod = 1. use counter skew function
#write code to check reconstruction for spherical
#loop reconstructing sphere until reasonable tmod is found