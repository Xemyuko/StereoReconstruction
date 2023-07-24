# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 10:48:46 2023

@author: myuey
"""
import numpy as np
import scripts as scr
import matplotlib.pyplot as plt

sphere_folder = "./test_data/Sphere230718/"

sphere_matrices_dir = "matrices/"

sphere_left = "c1/"
sphere_right = "c2/"

sphere_L, sphere_R = scr.load_first_pair(sphere_folder+sphere_left, sphere_folder+sphere_right)

xOffsetL = 1600
xOffsetR = 300
yOffsetT = 900
yOffsetB = 1100


F1 = scr.find_f_mat(sphere_L,sphere_R)
F2 = np.loadtxt(sphere_folder + sphere_matrices_dir + "f.txt", skiprows=2, delimiter = " ")
F3 = scr.find_f_mat(sphere_L,sphere_R, lmeds_mode = False)

rectL1,rectR1, H1, H2 = scr.rectify_pair(sphere_L,sphere_R, F1)
rectL2,rectR2, H1, H2 = scr.rectify_pair(sphere_L,sphere_R, F2)
rectL3,rectR3, H1, H2 = scr.rectify_pair(sphere_L,sphere_R, F3)
'''
scr.display_stereo(sphere_L, sphere_R)
scr.display_stereo(rectL1, rectR1)
scr.display_stereo(rectL2, rectR2)
scr.display_stereo(rectL3, rectR3)
'''
fig = scr.create_stereo_offset_fig(rectL1,rectR1,xOffsetL,xOffsetR,yOffsetT,yOffsetB)
plt.show()