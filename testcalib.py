# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:38:55 2023

@author: Admin
"""

import scripts as scr
import matplotlib.pyplot as plt
import cv2
import numpy as np
left_cal_folder = "./test_data/schachbrett-reduc0/c1/"
right_cal_folder = "./test_data/schachbrett-reduc0/c2/"
calmtx_folder = "./cal_mtx/"
rows = 6
columns =10
world_scaling = 0.04
def calibrate_save():

    #run camera calibration code on images
    kL, kR, distL, distR, R, T, E, F = scr.calibrate_cameras(left_cal_folder, right_cal_folder, "", 
                                                               rows, columns, world_scaling)
    scr.fill_mtx_dir(calmtx_folder, kL, kR, F, E, distL, distR, R, T)
    
images1 = scr.load_imgs_1_dir(left_cal_folder, "")
imgA = images1[0]
plt.imshow(imgA)
plt.show()
imgAG = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
plt.imshow(imgAG)
plt.show()
thresh = int(imgAG.max()*0.6)
mask = np.ones_like(imgAG)
mask[imgAG < thresh] = 0 
resA = imgAG*mask
plt.imshow(resA)
plt.show()


#calibrate_save()
'''
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
'''