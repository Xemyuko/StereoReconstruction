# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:57:32 2023

@author: myuey
"""
import numpy as np
import cv2
import scripts as scr


def triangulate_midpoint(pts1,pts2,R,t,kL,kR):
    pass
def trian_rdir(pts1,pts2, R, t):
    res = []
    for i,j in zip(pts1,pts2):
        i = np.append(i,0.0)
        j = np.append(j,0.0)
        dif1 = R[0] - (j[0] * R[2])
        dif2 = R[1] - (j[1] * R[2])
        x3_1 = np.dot(dif1,t)/np.dot(dif1,i)
        x3_2 = np.dot(dif2,t)/np.dot(dif2,i)
        x3 = np.average((x3_1,x3_2))
        x1 = i[0]/x3
        x2 = i[1]/x3
        res.append((x1,x2,x3))
    return np.asarray(res)
#define data sources
folder_statue = "./test_data/statue/"
matrix_folder = "matrix_folder/"
left_folder = "camera_L/"
right_folder = "camera_R/"
input_data = "Rekonstruktion30.pcf"
t_mod = 0.416657633
#load data
kL, kR, r_vec, t_vec = scr.initial_load(1,folder_statue + matrix_folder)
t_vec2 = t_vec*t_mod
imgL,imgR = scr.load_images(folderL = folder_statue+left_folder, folderR = folder_statue+right_folder)
xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(folder_statue + input_data)
kL_inv = np.linalg.inv(kL)
kR_inv = np.linalg.inv(kR)
#compute fundamental matrix
pts1b,pts2b,colb, F = scr.feature_corr(imgL[0],imgR[0], thresh = 0.6)
#compute essential matrix
ess = np.transpose(kR) @ F @ kL

#extract rotation and translation matrices from essential matrix with opencv
R1,R2,t = cv2.decomposeEssentialMat(ess)
col_arr = scr.gen_color_arr_black(len(xy1))
tri_res0 = scr.triangulate_list(xy1,xy2, r_vec, t_vec, kL_inv, kR_inv)
scr.convert_np_ply(tri_res0, col_arr,"testBaseline0.ply")
tri_res1 = scr.triangulate_list(xy1,xy2, r_vec, t_vec2, kL_inv, kR_inv)
scr.convert_np_ply(tri_res1, col_arr,"testBaseline1.ply")
tri_res2 = scr.triangulate_list(xy1,xy2, R2, t, kL_inv, kR_inv)
scr.convert_np_ply(tri_res2, col_arr,"testAutogen0.ply")
tri_res3 = scr.triangulate_list(xy1,xy2, R2, t*t_mod, kL_inv, kR_inv)
scr.convert_np_ply(tri_res3, col_arr,"testAutogen1.ply")

tri_res4 = trian_rdir(xy1,xy2,r_vec,t_vec2)
scr.convert_np_ply(tri_res4,col_arr,"testA.ply" )