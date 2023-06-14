# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:57:32 2023

@author: myuey
"""
import numpy as np
import cv2
import scripts as scr
def trian_mid(pts1,pts2,R,t,kL,kR):
    res = []
    for i,j in zip(pts1,pts2):
         
        #extend 2D pts to 3D
    
        pL = np.append(i,1.0)
        pR = np.append(j,1.0)
        vL = kL_inv @ pL
        vR = R@(kR_inv @ pR) + t.T 
        vR = vR[0]
        
        uL = vL/np.linalg.norm(vL)
        uR = vR/np.linalg.norm(vR)
        uC = np.cross(uR, uL)
        uC/=np.linalg.norm(uC)
        #solve the system using numpy solve
        eqL = pR - pL
        eqL = np.reshape(eqL,(3,1))

        eqR = np.asarray([uL,-uR,uC]).T

        resx = np.linalg.solve(eqR,eqL)
        resx = np.reshape(resx,(1,3))[0]
        qL = uL * resx[0] + pL
        qR = uR * resx[1] + pR
        resp = (qL + qR)/2
        res.append(resp)
    return np.asarray(res)
def reverse_trian(pts1,pts2,R,t,kL,kR):
    res = []
    for i,j in zip(pts1,pts2):
        #extend 2D pts to 3D
  
        pL = np.append(i,1.0)
        pR = np.append(j,1.0)  
        vL = R.T@(kL_inv@pL) - t.T 
        vR = kR_inv@pR
        vL = vL[0]
        
        uL = vL/np.linalg.norm(vL)
        uR = vR/np.linalg.norm(vR)
        uC = np.cross(uR, uL)
        uC/=np.linalg.norm(uC)
        #solve the system using numpy solve
        eqL = pR - pL
        eqL = np.reshape(eqL,(3,1))

        eqR = np.asarray([uL,-uR,uC]).T

        resx = np.linalg.solve(eqR,eqL)
        resx = np.reshape(resx,(1,3))[0]
        qL = uL * resx[0] + pL
        qR = uR * resx[1] + pR
        resp = (qL + qR)/2
        res.append(resp)
    return np.asarray(res)
def avg_trian(pts1,pts2,R,t,kL,kR):
    res = []
    for i,j in zip(pts1,pts2):
         #extend 2D pts to 3D
   
         pL = np.append(i,1.0)
         pR = np.append(j,1.0)
         
         vLa = kL_inv @ pL
         vRa = R@(kR_inv @ pR) + t.T 
         vRa = vRa[0]
         
         uLa = vLa/np.linalg.norm(vLa)
         uRa = vRa/np.linalg.norm(vRa)
         uCa = np.cross(uRa, uLa)
         uCa/=np.linalg.norm(uCa)
         #solve the system using numpy solve
         eqLa = pR - pL
         eqLa = np.reshape(eqLa,(3,1))

         eqRa = np.asarray([uLa,-uRa,uCa]).T

         resxa = np.linalg.solve(eqRa,eqLa)
         resxa = np.reshape(resxa,(1,3))[0]
         qLa = uLa * resxa[0] + pL
         qRa = uRa * resxa[1] + pR
         respa = (qLa + qRa)/2
         
         vL = R.T@(kL_inv@pL) - t.T 
         vR = kR_inv@pR
         vL = vL[0]
         
         uL = vL/np.linalg.norm(vL)
         uR = vR/np.linalg.norm(vR)
         uC = np.cross(uR, uL)
         uC/=np.linalg.norm(uC)
         #solve the system using numpy solve
         eqL = pR - pL
         eqL = np.reshape(eqL,(3,1))

         eqR = np.asarray([uL,-uR,uC]).T

         resx = np.linalg.solve(eqR,eqL)
         resx = np.reshape(resx,(1,3))[0]
         qL = uL * resx[0] + pL
         qR = uR * resx[1] + pR
         resp = (qL + qR)/2
         resn = (resp+respa)/2
         n = 2
         resn *= np.asarray([1,1,n])
         res.append(resn)
    return np.asarray(res)
#define data sources
folder_statue = "./test_data/statue/"
matrix_folder = "matrix_folder/"
left_folder = "camera_L/"
right_folder = "camera_R/"
input_data = "Rekonstruktion30.pcf"
t_mod = 0.416657633
t_mod2 = 1-t_mod
#load data
kL, kR, r_vec, t_vec = scr.initial_load(1,folder_statue + matrix_folder)
t_vec2 = t_vec*t_mod
t_vec3 = t_vec*t_mod2
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

pL = pts1b[0]
pR = pts2b[0]
kL_inv = np.linalg.inv(kL)
kR_inv = np.linalg.inv(kR)
#test triangulation functions
'''
test1 = trian_mid(xy1,xy2,r_vec,t_vec, kL, kR)
scr.convert_np_ply(test1,col_arr,"t1.ply")
test2 = reverse_trian(xy1,xy2,r_vec,t_vec, kL, kR)
scr.convert_np_ply(test2,col_arr,"t2.ply")
'''
test3 = avg_trian(xy1,xy2,r_vec,t_vec, kL, kR)
scr.convert_np_ply(test3,col_arr,"t3.ply")