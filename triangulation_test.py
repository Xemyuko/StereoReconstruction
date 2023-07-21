# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:57:32 2023

@author: myuey
"""
import numpy as np
import cv2
import scripts as scr
def avg_trian(pts1,pts2,R,t,kL,kR):
    res = []
    for i,j in zip(pts1,pts2):
         #extend 2D pts to 3D
   
         pL = np.append(i,1.0)
         pR = np.append(j,1.0)
         
         vLa = kL_inv @ pL
         amat = np.linalg.inv(kR @ R)
         vRa = amat @ pR  + t.T 
        # vRa = R@(kR_inv @ pR) + t.T 
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
         
       #  vL = R.T@(kL_inv@pL) - t.T 
         bmat = np.linalg.inv(kL @ R.T)
         vL = bmat @ pL - t.T
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
        
         res.append(resn)
    return np.asarray(res)

def scale_trian(pts1,pts2,R,t,kL,kR):
    res = []
    kL_inv = np.linalg.inv(kL)
    kR_inv = np.linalg.inv(kR)
    for i,j in zip(pts1,pts2):
         #extend 2D pts to 3D
   
         pL = np.append(i,1.0)
         pR = np.append(j,1.0)
         
         vL = kL_inv @ pL
         vL = vL/np.linalg.norm(vL)
         vR = R@(kR_inv @ pR)
         vR = vR/np.linalg.norm(vR)
         
         ang_A = np.arccos(np.clip(np.dot(vL, t/np.linalg.norm(t)), -1.0, 1.0))
         ang_B = np.arccos(np.clip(np.dot(vR, t/np.linalg.norm(t)), -1.0, 1.0))
         
         facA = np.sin(ang_B)/np.sin(np.pi-ang_A-ang_B)
         facB = np.sin(ang_B)/np.sin(np.pi-ang_A-ang_B)
         
         resx = (facA*vL + facB*vR)/2
         res.append(resx)
    return np.asarray(res) 

def tri_no_skew(pts1,pts2,r_vec,t_vec,kL,kR):
    res = []
    kL_inv = np.linalg.inv(kL)
    kR_inv = np.linalg.inv(kR)
    t_vec = t_vec.T[0]
    
    for i,j in zip(pts1,pts2):
         #extend 2D pts to 3D
   
         p1 = np.append(i,1.0)
         p2 = np.append(j,1.0)
         v1 = kL_inv @ p1
         v2 = r_vec@(kR_inv @ p2) + t_vec
         #Calculate distances along each vector for closest points, then sum and halve for midpoints
         phi = (t_vec[0]-v1[0]*t_vec[2])/(v1[0]*v2[2]-v2[0])
         
         lam = t_vec[2]+phi*v2[2]
         
         res1 = np.asarray([(lam*v1[0]+phi*v2[0])/2,(lam*v1[1]+phi*v2[1])/2,(lam*v1[2]+phi*v2[2])/2])
         
         v3 = r_vec.T@(kR_inv @ p2) - t_vec
         v4 = kR_inv @ p2
         v3=v3.T
         
         phi2 = (t_vec[0]-v3[0]*t_vec[2])/(v3[0]*v4[2]-v4[0])
         
         lam2 = t_vec[2]+phi2*v4[2]
         
         res2 = np.asarray([(lam2*v3[0]+phi2*v4[0])/2,(lam2*v3[1]+phi2*v4[1])/2,(lam2*v3[2]+phi2*v4[2])/2])
         
         resx = (res1 + res2)/2
         res.append(resx)
    return np.asarray(res)

def tri2(pts1,pts2,r_vec,t_vec,kL,kR):
    res = []
    kL_inv = np.linalg.inv(kL)
    t_vec = t_vec.T[0]
    for i,j in zip(pts1,pts2):
         #extend 2D pts to 3D
         p1 = np.append(i,1.0)
         p2 = np.append(j,1.0)
         v1 = kL_inv @ p1
         #v1 = v1/np.linalg.norm(v1)

         a_mat = np.linalg.inv(kR @ r_vec)
         v2 = a_mat @ p2
         #v2 = v2/np.linalg.norm(v2)

         n_vek = np.cross(v1, np.cross(v1,v2))
         m_vek = np.cross(v2, np.cross(v1,v2))


         q1 = t_vec + (n_vek  @ -t_vec)/(n_vek @ v2) * v2
         q2 = (m_vek @ t_vec)/(m_vek @ v1) * v1

         resq = (q1 + q2)/2
         res.append(resq)
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
test_tri2 = tri2(xy1,xy2,r_vec,t_vec2, kL, kR)
scr.convert_np_ply(test_tri2,col_arr,"t2.ply")