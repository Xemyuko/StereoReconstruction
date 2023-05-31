# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:57:32 2023

@author: myuey
"""
import numpy as np
import cv2
import scripts as scr

def trian_mid3(pts1,pts2,R,t,kL,kR):
    res = []
    for i,j in zip(pts1,pts2):
        #extend 2D pts to 3D
        pL = np.append(i,1.0)
        j = np.append(j,1.0)
        #apply transformation to right-camera point
        j = R @ j + t.T 
        pR=j[0]
        #define focal points as second-point definition of lines
        vzL = (kL[0,0] + kL[1,1])/2
        vL = np.asarray([kL[0,2],kL[1,2],vzL])
        vzR = (kR[0,0] + kR[1,1])/2
        vR = R@np.asarray([kR[0,2],kR[1,2],vzR])+t.T
        vR = vR[0]
        #calculate slopes of lines
        distL = np.sqrt((vL[0]-pL[0])**2 + (vL[1]-pL[1])**2 + (vL[2]-pL[2])**2)
        distR = np.sqrt((vR[0]-pR[0])**2 + (vR[1]-pR[1])**2 + (vR[2]-pR[2])**2)
        dL = np.asarray([vL[0]-pL[0],vL[1]-pL[1],vL[2]-pL[2]])/distL
        dR = np.asarray([vR[0]-pR[0],vR[1]-pR[1],vR[2]-pR[2]])/distR
        #breakdown of points
        x1,y1,z1 = pL
        x2,y2,z2 = vL
        x3,y3,z3 = pR
        x4,y4,z4 = vR
        #breakdown of slopes
        xd1,yd1,zd1 = dL
        xd2,yd2,zd2 = dR
        #breakdown of initial points difference
        xd3,yd3,zd3 = pR-pL
        #slope scales
        rdL = xd1**2 + yd1**2 + zd1**2
        rdR = xd2**2 + yd2**2 + zd2**2
        #dot products
        dot1 = xd1*xd2 + yd1*yd2 + zd1*zd2
        dot2 = xd1*xd3 + yd1*yd3 + zd1*zd3
        dot3 = xd3*xd2 + yd3*yd2 + zd3*zd2
        # rdL*s - dot1*sb - dot2= 0
        # dot1*s - R2*sb - dot3 = 0
        #rearrange and solve for distances to closest points
        den = dot1**2 - rdL*rdR
        s = (dot1*dot3- rdR*dot2)/den
        sb = (rdL*dot3 - dot1*dot2)/den
        #apply distances to line equations
        ps = pL + s*dL
        pt = pR + sb*dR
        #midpoint
        resx = (ps + pt)/2
        res.append(resx)
    return np.asarray(res)

def trian_mid2(pts1,pts2,R,t,kL,kR):
    res = []
    for i,j in zip(pts1,pts2):
        #extend 2D pts to 3D
        pL = np.append(i,1.0)
        j = np.append(j,1.0)
        #apply transformation to right-camera point
        j = R @ j + t.T 
        pR=j[0]
        #define focal points as second-point definition of lines
        vzL = (kL[0,0] + kL[1,1])/2
        vL = np.asarray([kL[0,2],kL[1,2],vzL])
        vzR = (kR[0,0] + kR[1,1])/2
        vR = R@np.asarray([kR[0,2],kR[1,2],vzR])+t.T
        vR = vR[0]
        distL = np.sqrt((vL[0]-pL[0])**2 + (vL[1]-pL[1])**2 + (vL[2]-pL[2])**2)
        distR = np.sqrt((vR[0]-pR[0])**2 + (vR[1]-pR[1])**2 + (vR[2]-pR[2])**2)
        dL = np.asarray([vL[0]-pL[0],vL[1]-pL[1],vL[2]-pL[2]])/distL
        dR = np.asarray([vR[0]-pR[0],vR[1]-pR[1],vR[2]-pR[2]])/distR
        #compute unit vectors in directions lines
        uL = (vL - pL)/np.linalg.norm(vL-pL)
        uR = (vR - pR)/np.linalg.norm(vR-pR)
        #find the unit direction vector for the line normal to both camera lines
        uC = np.cross(uR, uL)
        uC/=np.linalg.norm(uC)
        #solve the system using numpy solve
        eqL = pR - pL
        eqL = np.reshape(eqL,(3,1))

        eqR = np.asarray([uL,-uR,uC]).T

        resx = np.linalg.solve(eqR,eqL)
        resx = np.reshape(resx,(1,3))[0]
        qL = resx[0]*dL + pL
        qR = resx[1]*dR + pR
        resp = (qL + qR)/2
        res.append(resp)
    return np.asarray(res)
def trian_mid1(pts1,pts2,R,t,kL,kR):
    res = []
    for i,j in zip(pts1,pts2):
        #extend 2D pts to 3D
        pL = np.append(i,1.0)
        j = np.append(j,1.0)
        #apply transformation to right-camera point
        j = R @ j + t.T 
        pR=j[0]
        #compute line factor d
        vzL = (kL[0,0] + kL[1,1])/2
        vL = np.asarray([kL[0,2],kL[1,2],vzL])
        vzR = (kR[0,0] + kR[1,1])/2
        vR = R@np.asarray([kR[0,2],kR[1,2],vzR])+t.T
        vR = vR[0]
        distL = np.sqrt((vL[0]-pL[0])**2 + (vL[1]-pL[1])**2 + (vL[2]-pL[2])**2)
        distR = np.sqrt((vR[0]-pR[0])**2 + (vR[1]-pR[1])**2 + (vR[2]-pR[2])**2)
        dL = np.asarray([vL[0]-pL[0],vL[1]-pL[1],vL[2]-pL[2]])/distL
        dR = np.asarray([vR[0]-pR[0],vR[1]-pR[1],vR[2]-pR[2]])/distR

        n = np.cross(dL,dR)
        nL = np.cross(dL, n)
        nR = np.cross(dR,n)
        
        nL = nL/np.linalg.norm(nL)
        nR = nR/np.linalg.norm(nR)
        
        cL = pL + (np.dot((pR - pL),nR)/np.dot(dL,nR))*dL
        cR = pR + (np.dot((pL - pR),nL)/np.dot(dR,nL))*dR
        m = (cL + cR)/2
        res.append(m)
    return np.asarray(res)
def trian_rdir(pts1,pts2, R, t):
    res = []
    for i,j in zip(pts1,pts2):
        i = np.append(i,1.0)
        j = np.append(j,1.0)
        j = R @ j + t.T
        j = j[0]
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


#test triangulation functions
test1 = trian_mid3(xy1,xy2,r_vec,t_vec, kL, kR)
scr.convert_np_ply(test1,col_arr,"t3.ply")