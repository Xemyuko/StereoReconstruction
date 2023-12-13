# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:09:26 2023

@author: Admin
"""

import numpy as np
import scripts as scr
import matplotlib.pyplot as plt
import ncc_core as ncc
import confighandler as ch
from tqdm import tqdm
import cv2
import json
import numba


float_epsilon = 1e-9


def test_fix2():
    #Load Matrices
    testFolder = "./test_data/maustest/"
    input_data = "Maus1.pcf"
    skiprow = 2
    delim = ' '
    kL = np.loadtxt(testFolder + 'Kl.txt', skiprows=skiprow, delimiter = delim)
    kR = np.loadtxt(testFolder +  'Kr.txt', skiprows=skiprow, delimiter = delim)
    kL_inv = np.linalg.inv(kL)
    kR_inv = np.linalg.inv(kR)
    r_vec = np.loadtxt(testFolder +  'R.txt', skiprows=skiprow, delimiter = delim)
    t_vec = np.loadtxt(testFolder +  't.txt', skiprows=skiprow, delimiter = delim)
    #Access 2D points from reference pcf
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(testFolder + input_data)
    res=[]
    for i,j in zip(xy1,xy2):
        res.append(scr.triangulate_avg(i,j,r_vec,t_vec, kL_inv, kR_inv ))
    res = np.asarray(res)
    scr.create_ply(res, "testmaus2")
#test_fix2()
def test_fix():
    #Load Matrices
    testFolder = "./test_data/maustest/"
    input_data = "Maus1.pcf"
    skiprow = 2
    delim = ' '
    kL = np.loadtxt(testFolder + 'Kl.txt', skiprows=skiprow, delimiter = delim)
    kR = np.loadtxt(testFolder +  'Kr.txt', skiprows=skiprow, delimiter = delim)
    
    r_vec = np.loadtxt(testFolder +  'R.txt', skiprows=skiprow, delimiter = delim)
    t_vec = np.loadtxt(testFolder +  't.txt', skiprows=skiprow, delimiter = delim)
    #Create calc matrices 

    k1 = np.c_[kL, np.asarray([[0],[0],[1]])]
    
    k2 = np.c_[kR, np.asarray([[0],[0],[1]])]
    
    RT = np.c_[r_vec, t_vec]
    RT = np.r_[RT, [np.asarray([0,0,0,1])]]
    
    P1 = k1 @ np.eye(4,4)
    
    P2 = k2 @ RT
    #Access 2D points from reference pcf
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(testFolder + input_data)
    res=[]
    cor_thresh = 0.6
    ignore_thresh = True
    for i,j,k in zip(xy1,xy2, correl):
        #Check correlation threshold
        
        if(k >= cor_thresh or ignore_thresh):
            #Create solution matrix
            sol0 = i[0] * P1[2,:] - P1[0,:]
            sol1 = i[1] * P1[2,:] - P1[1,:]
            sol2 = j[0] * P2[2,:] - P2[0,:]
            sol3 = j[1] * P2[2,:] - P2[1,:]
            
            solMat = np.stack((sol0,sol1,sol2,sol3))
            
            #Apply SVD to solution matrix to find triangulation
            U,s,vh = np.linalg.svd(solMat,full_matrices = True)
            vh = vh.T
            Q = vh[:,3]

            Q *= 1/Q[3]
    
            res.append(Q[:3])
    res = np.asarray(res)
    filt_res = []
    filt_thresh = 0.6
    for i,j in zip(geom_arr,correl):
        if(j >= filt_thresh):
            filt_res.append(i)
    filt_res = np.asarray(filt_res)
    scr.create_ply(res, "testmaus")
    scr.create_ply(geom_arr, "referencemaus")
    scr.create_ply(filt_res, 'filtmaus')
#test_fix()
def t1():
    folder = "./test_data/calibObjects/"
    filename = folder + 'testconeread.json'
    f = open(filename)
    data = json.load(f)
    vertices = np.asarray(data['objects'][0]['vertices'])
    f.close()
    maxVals = np.max(vertices, axis = 0)
    minVals = np.min(vertices, axis = 0)
    refXdist = maxVals[0] - minVals[0]
    refYdist = maxVals[2] - minVals[2]
    refZdist = maxVals[1] - minVals[1]
    chkZdist = 40
    print(data['objects'][0])
  

def testfreq():
    pass
    #load image data
    #load matrices
    #apply stereo rectification to images
    #apply fft to images
    #run ncc on transforms
    #inverse fft
    #check results
    
def compare_cor(res_list, entry_val, threshold, recon = True):
    #duplicate comparison and correlation thresholding, run when trying to add points to results
    remove_flag = False
    pos_remove = 0
    entry_flag = False
    counter = 0
    if(recon):
        if(entry_val[1] < 0 or entry_val[2] < threshold):
            return pos_remove,remove_flag,entry_flag
    else:
        if(entry_val[1] < 0):
            return pos_remove,remove_flag,entry_flag
    for i in range(len(res_list)):       
        
        if(res_list[i][1] == entry_val[1] and res_list[i][3][0] - entry_val[3][0] < float_epsilon and
           res_list[i][3][1] - entry_val[3][1] < float_epsilon):
            #duplicate found, check correlation values and mark index for removal
            remove_flag = (res_list[i][2] > entry_val[2])
            pos_remove = i
            break
        else:
            counter+=1
    #end of list reached, no duplicates found, entry is valid
    if(counter == len(res_list)):
        entry_flag = True
    return pos_remove,remove_flag,entry_flag 

@numba.jit(nopython=True)
def cor_pix_norect(Gi,n, xLim, maskR, xOffset1, xOffset2,yOffset1, yOffset2):
    max_cor = 0.0
    max_index = -1
    max_mod = [0,0] #default to no change
    agi = np.sum(Gi)/n
    val_i = np.sum((Gi-agi)**2)
    for xi in range(xOffset1, xLim-xOffset2):
        for yi in range(xOffset1, xLim-xOffset2):
            Gt = maskR[:,yi,xi]
            agt = np.sum(Gt)/n        
            val_t = np.sum((Gt-agt)**2)
            if(val_i > float_epsilon and val_t > float_epsilon): 
                cor = np.sum((Gi-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))              
                if cor > max_cor:
                    max_cor = cor
                    max_index = xi
def test_fmat_ncc():
    #load image data
    folderD = './test_data/testsphere2/'
    folderL = folderD + 'camL/'
    folderR = folderD + 'camR/'
    imgL,imgR = scr.load_images(folderL,folderR)
    #Apply masks to images
    thresh_val = 30
    avgL = np.asarray(imgL).mean(axis=(0))
    avgR = np.asarray(imgR).mean(axis=(0))
    maskL = scr.mask_avg_list(avgL,imgL, thresh_val)
    maskR = scr.mask_avg_list(avgR,imgR, thresh_val)

    maskL = np.asarray(maskL)
    maskR = np.asarray(maskR)
    #load matrices
    folderM = folderD + 'Matrix_folder/'
    kL, kR, r, t = scr.load_mats(folderM)
    #drastically reduce total number of image points in left image
    #get image shape
    imshape = imgL[0].shape
    #Take every 400th point stack and add to list
    pointsL = []
    for i in range(0,imshape[0],400):
        for j in range(0,imshape[1],400):
            pointsL.append(maskL[:,i,j])
    pointsL = np.asarray(pointsL)
    #apply ncc to match these points in right image, but cannot use stereo rectification.
    xLim = imshape[1]
    yLim = imshape[0]
    xOffsetL = 1
    xOffsetR = 1
    yOffsetT = 1
    yOffsetB = 1
    n = len(imgL)
    thresh = 0.8
    res_ent = []
    res_y = []
    for a in pointsL:
        if(np.sum(a) != 0): #dont match fully dark slices
            x_match,cor_val,subpix = cor_pix_norect(a,n, xLim, maskR, xOffsetL, xOffsetR)
                
            pos_remove, remove_flag, entry_flag = compare_cor(res_y,
                                                              [a[0],x_match, cor_val, subpix, a[1]], thresh)
            if(remove_flag):
                res_y.pop(pos_remove)
                res_y.append([a[0],x_match, cor_val, subpix, a[1]])
            elif(entry_flag):
                res_y.append([a[0],x_match, cor_val, subpix, a[1]])
    res_ent.append(res_y)
    #use the resulting point matches to compute the fundamental matrix
test_fmat_ncc()    
    
