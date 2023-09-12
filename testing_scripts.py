# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:09:26 2023

@author: Admin
"""

import numpy as np
import scripts as scr
import scipy.linalg as sclin
def t1():
    #define data sources
    folder_statue = "./test_data/statue/"
    matrix_folder = "matrix_folder/"
    left_folder = "camera_L/"
    right_folder = "camera_R/"
    input_data = "Rekonstruktion30.pcf"
    #load data
    kL, kR, R, t = scr.initial_load(1,folder_statue + matrix_folder)
    imgL,imgR = scr.load_images(folderL = folder_statue+left_folder, folderR = folder_statue+right_folder)
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(folder_statue + input_data)
    #F = scr.find_f_mat(imgL[0],imgR[0])
    kL_inv = np.linalg.inv(kL)
    kR_inv = np.linalg.inv(kR)
    
    ind = 59
    
    pL = xy1[ind]
    pR = xy2[ind]
    print('P________')
    print(pL)
    print(pR)

    projR = np.append(R,t,axis = 1) 
    projL = np.append(np.identity(3),np.asarray([[0],[0],[0]]),axis = 1)
    A_L = kL @ projL
    A_R = kR @ projR
    r0 = pL[1]*A_L[2,:] - A_L[1,:]
    r1 = A_L[0,:] - pL[0]*A_L[2,:]
    r2 = pR[1]*A_R[2,:] - A_R[1,:]
    r3 = A_R[0,:] - pR[0]*A_R[2,:]
    somat = np.vstack((r0,r1,r2,r3))
    print('S________')
    print(somat)
    
    
    u, s, vh = np.linalg.svd(somat, full_matrices = True)
    M, N = u.shape[0], vh.shape[1]
    rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()

    Q *= 1/Q[3]
    print('Q________')
    print(Q)
    print('A________')
    print(geom_arr[ind])
    pL = np.append(pL,1.0)
    pR = np.append(pR,1.0)
    
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
    print('R__________')
    print(resn)
t1()