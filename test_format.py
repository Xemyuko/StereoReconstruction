# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 10:48:46 2023

@author: myuey
"""
import numpy as np
import scripts as scr
from tqdm import tqdm
import matplotlib.pyplot as plt

sphere_folder = "./test_data/Sphere230718/"

sphere_matrices_dir = "matrices/"

sphere_left = "c1/"
sphere_right = "c2/"

sphere_L, sphere_R = scr.load_first_pair(sphere_folder+sphere_left, sphere_folder+sphere_right)



F1 = scr.find_f_mat(sphere_L,sphere_R, precise = False)
F2 = np.loadtxt(sphere_folder + sphere_matrices_dir + "f.txt", skiprows=2, delimiter = " ")
F3 = scr.find_f_mat(sphere_L,sphere_R, precise = False, lmeds_mode = False)

rectL1,rectR1 = scr.rectify_lists(sphere_L,sphere_R, F1)
rectL2,rectR2 = scr.rectify_lists(sphere_L,sphere_R, F2)
rectL3,rectR3 = scr.rectify_lists(sphere_L,sphere_R, F3)

scr.display_stereo(sphere_L[0], sphere_R[0])
scr.display_stereo(rectL1[0], rectR1[0])
scr.display_stereo(rectL2[0], rectR2[0])
scr.display_stereo(rectL3[0], rectR3[0])
