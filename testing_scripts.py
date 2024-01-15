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
import time
import os
import matplotlib.pyplot as plt
float_epsilon = 1e-9

def remove_z_outlier(geom_arr, col_arr):
    ind_del = []
    for i in range(geom_arr.shape[0]):

        if geom_arr[i,2] > np.max(geom_arr[:,2])*0.8:
            ind_del.append(i)

    geom_arr = np.delete(geom_arr, np.asarray(ind_del), axis = 0)
    col_arr = np.delete(col_arr,np.asarray(ind_del), axis = 0 )
    return geom_arr, col_arr
def remove_z_outlier_no_col(geom_arr):
    ind_del = []
    for i in range(geom_arr.shape[0]):

        if geom_arr[i,2] > np.max(geom_arr[:,2])*0.8:
            ind_del.append(i)

    geom_arr = np.delete(geom_arr, np.asarray(ind_del), axis = 0)

    return geom_arr
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
def cor_acc_linear_grid(Gi,x,y,n, xLim, maskR, xOffset1, xOffset2, interp_num):
    max_cor = 0
    max_index = -1
    max_mod = [0.0,0.0] #default to no change
    agi = []
    val_i = []
    for a in range(len(Gi)):
        agi.append(np.sum(Gi[a])/n)
        val_i.append(np.sum((Gi[a]-agi[a])**2))
    #Search the entire line    
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = [maskR[:,y,xi],maskR[:,y-1,xi],maskR[:,y+1,xi],maskR[:,y,xi-1],maskR[:,y,xi+1]]
        agt = []
        val_t = []
        for b in range(len(Gt)):
            agt.append(np.sum(Gt[b])/n)
            val_t.append(np.sum((Gt[b]-agt[b])**2))
        cor_arr = []
        for c in range(len(Gt)):
            if(val_i[c] > float_epsilon and val_t[c] > float_epsilon):
                cor_arr.append(np.sum((Gi[c]-agi[c])*(Gt[c] - agt[c]))/(np.sqrt(val_i[c]*val_t[c])))
            else:
                cor_arr.append(0)
        cor_arr = np.asarray(cor_arr)
        cor_check = np.mean(cor_arr)
        if cor_check > max_cor:
            max_cor = cor_check
            max_index = xi
    #search around the found best index and do linear interpolation
    if(max_index > -1):  
        #define increment of interpolation
        increment = 1/ (interp_num + 1)
        #define changes to get 8-neighbors of the selected point
        #[N,S,W,E]
        coord_card = [(-1,0),(1,0),(0,-1),(0,1)]
        #[NW,SE,NE,SW]
        coord_diag = [(-1,-1),(1,1),(-1,1),(1,-1)]
        #define points to look at
        G_cores =  np.asarray([maskR[:,y,max_index],maskR[:,y-1,max_index],maskR[:,y+1,max_index],maskR[:,y,max_index-1],
                          maskR[:,y,max_index+1],maskR[:,y-1,max_index-1],maskR[:,y+1,max_index+1],maskR[:,y-1,max_index+1],
                                            maskR[:,y+1,max_index-1]])
        #define order of surrounding point coordinate modifiers
        coords = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
        #define base coordinates of points to look at
        G_coords = [(y,max_index),(y-1,max_index),(y+1,max_index),(y,max_index-1),
                    (y,max_index+1),(y-1,max_index-1),(y+1,max_index+1),(y-1,max_index+1),(y+1,max_index-1)]
        G_total = []
        for i,loc in zip(G_cores,G_coords):#loop through central points and their coordinates
            G_line = [i] #initialize per-central-point list with central point
            for j in coords:#loop through coordinate modifiers
                G_line.append(maskR[:,loc[0]+j[0],loc[1]+j[1]])#add coordinate modifiers to central point and save result
            
        #Need to access the 4-neighbors of each of the cardinal and diagonal points
        #define loops
        #check cardinal
        G_N = [maskR[:,y+1,max_index]]

        
        for i in range(len(coord_card)):
            val = G_card[i] - G_cent
            if(i<2):
                G_check = G_card[i]
                ag_check = np.sum(G_check)/n
                val_check = np.sum((G_check-ag_check)**2)
                if(val_i > float_epsilon and val_check > float_epsilon): 
                    cor = np.sum((Gi-agi)*(G_check - ag_check))/(np.sqrt(val_i*val_check))     
                    if cor > max_cor:
                        max_cor = cor
                        max_mod = max_mod+[coord_card[i][0]*1.0, coord_card[i][1]*1.0]
            for j in range(interp_num):
                G_check= ((j+1)*increment * val) + G_cent
                ag_check = np.sum(G_check)/n
                val_check = np.sum((G_check-ag_check)**2)
                if(val_i > float_epsilon and val_check > float_epsilon): 
                    cor = np.sum((Gi-agi)*(G_check - ag_check))/(np.sqrt(val_i*val_check))
                     
                    if cor > max_cor:
                        max_cor = cor
                        max_mod = max_mod+[coord_card[i][0]*(j+1)*increment,coord_card[i][1]*(j+1)*increment]
                        
        #check diagonal
        diag_len = 1.41421356237 #sqrt(2), possibly faster to just have this hard-coded
        for i in range(len(coord_diag)):
            val = G_diag[i] - G_cent
            for j in range(interp_num):
                G_check= (((j+1)*increment * val)/diag_len) + G_cent
                ag_check = np.sum(G_check)/n
                val_check = np.sum((G_check-ag_check)**2)
                if(val_i > float_epsilon and val_check > float_epsilon): 
                    cor = np.sum((Gi-agi)*(G_check - ag_check))/(np.sqrt(val_i*val_check))
                         
                    if cor > max_cor:
                        max_cor = cor
                        max_mod = max_mod+[coord_diag[i][0]*(j+1)*increment,coord_diag[i][1]*(j+1)*increment]      
    return max_index,max_cor,max_mod
def grid_cor(config):
    kL, kR, r_vec, t_vec, kL_inv, kR_inv, F, imgL, imgR, imshape, maskL, maskR = ncc.startup_load(config, True)    
    #define constants for window
    xLim = imshape[1]
    yLim = imshape[0]
    xOffsetL = config.x_offset_L
    xOffsetR = config.x_offset_R
    yOffsetT = config.y_offset_T
    yOffsetB = config.y_offset_B
    thresh = config.thresh
    interp = config.interp
    rect_res = []
    n = len(imgL)
    interval = 1
    for y in range(yOffsetT, yLim-yOffsetB):
        res_y = []
        for x in range(xOffsetL, xLim-xOffsetR, interval):
            Gi = [maskL[:,y,x],maskL[:,y-1,x],maskL[:,y+1,x],maskL[:,y,x-1],maskL[:,y,x+1]]
            if(np.sum(Gi) != 0): #dont match fully dark slices
                x_match,cor_val,subpix = cor_acc_linear_grid(Gi,x,y,n, xLim, maskR, xOffsetL, xOffsetR, interp)
                    
                pos_remove, remove_flag, entry_flag = compare_cor(res_y,
                                                                  [x,x_match, cor_val, subpix, y], thresh)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, subpix, y])
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, subpix, y])
        rect_res.append(res_y)
def test_grid_cor():
    pass
    #Create config object
    #run current subpix ncc correlation method to get a baseline of number of correlated points
    #create pointcloud for baseline reference
    #create xyz datafile
    #Compare result with gridcor, which checks surroundings as well for more information in the match
    #create pointcloud for gridcor to check
    #create xyz datafile
    
@numba.jit(nopython=True)
def spat_cor_lin(Gi,x,y,n, xLim, img_cR, xOffset1, xOffset2, interp_num, coord_mods):
    max_cor = 0
    max_index = -1
    max_mod = [0.0,0.0] #default to no change
    agi = []
    val_i = []
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = []
        Gt.append(img_cR[y,xi])
        for a in coord_mods:
            Gt.append(img_cR[y+a[0],xi+a[-1]])
        agt = np.sum(Gt)/n        
        val_t = np.sum((Gt-agt)**2)
        if(val_i > float_epsilon and val_t > float_epsilon): 
            cor = np.sum((Gi-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))              
            if cor > max_cor:
                max_cor = cor
                max_index = xi
                
    #search around the found best index
    if(max_index > -1):  
        increment = 1/ (interp_num + 1)
         
        #define points
        G_cent =  img_cR[y,max_index]
        for a in coord_mods:
            G_cent.append(img_cR[y+a[0],max_index+a[-1]])
        #[N,S,W,E]
        coord_card = [(-1,0),(1,0),(0,-1),(0,1)]
        #[NW,SE,NE,SW]
        coord_diag = [(-1,-1),(1,1),(-1,1),(1,-1)]
        G_card = [maskR[:,y-1,max_index],maskR[:,y+1,max_index],maskR[:,y,max_index-1],
                          maskR[:,y,max_index+1]]
        G_diag = [maskR[:,y-1,max_index-1],maskR[:,y+1,max_index+1],maskR[:,y-1,max_index+1],
                          maskR[:,y+1,max_index-1]]
def spat_cor(config):
    kL, kR, r_vec, t_vec, kL_inv, kR_inv, F, imgL, imgR, imshape, maskL, maskR = ncc.startup_load(config, True)    
    #define constants for window
    xLim = imshape[1]
    yLim = imshape[0]
    xOffsetL = config.x_offset_L
    xOffsetR = config.x_offset_R
    yOffsetT = config.y_offset_T
    yOffsetB = config.y_offset_B
    thresh = config.thresh
    interp = config.interp
    rect_res = []
    n = len(imgL)
    interval = 1
    img_cL = maskL[0]
    img_cR = maskR[0]
    coord_mods = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    for y in  tqdm(range(yOffsetT, yLim-yOffsetB)):
        res_y = []
        for x in range(xOffsetL, xLim-xOffsetR, interval):
            Gi = []
            Gi.append(img_cL[y,x])
            for a in coord_mods:
                Gi.append(img_cL[y+a[0],x+a[-1]])
            x_match,cor_val,subpix = spat_cor_lin(Gi,x,y,n, xLim, img_cR, xOffsetL, xOffsetR, interp, coord_mods)
def spat_test():
    #Create config object
    #load first pair of images
    #load matrices
    #rectify image pair
    #create function for grid searching - access 8 neighbor of a given point
    #create copy of 
    pass

def test_single_folder_load_images(folder, imgLInd, imgRInd, ext):
    imgL = []
    imgR = [] 
    resL = []
    resR = []
    #Access and store all files with the image extension given
    imgFull = []
    for file in os.listdir(folder):
        if file.endswith(ext):
            imgFull.append(file)
    #Sort images into left and right based on if they contain the respective indicators
    #if they do not have either, ignore them
     
    for i in imgFull:
        if imgLInd in i:
            resL.append(i)
        elif imgRInd in i:
            resR.append(i)      
    #sort left and right images
    resL.sort()
    resR.sort()
    for i in resL:
        img = plt.imread(folder + i)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgL.append(img)
    for i in resR:
        img = plt.imread(folder + i)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgR.append(img)
    return np.asarray(imgL),np.asarray(imgR)
def run_test1():
    folder = "./test_data/testsphere2/combined_img/"
    imgLInd = "cam1"
    imgRInd = "cam2"
    ext = ".jpg"
    imgL,imgR = test_single_folder_load_images(folder, imgLInd, imgRInd, ext)
    print(imgL.shape,imgR.shape)
    plt.imshow(imgL[0])
    plt.show()
    plt.imshow(imgR[0])
    plt.show()
print(np.asarray([]).shape)
def testfreq():
    pass
    #load image data
    #load matrices
    #apply stereo rectification to images
    #apply fft to images
    #run ncc on transforms
    #inverse fft
    #check results
    
def compare_cor_fmat(res_list, entry_val, threshold):
    #duplicate comparison and correlation thresholding, run when trying to add points to results
    remove_flag = False
    pos_remove = 0
    entry_flag = False
    counter = 0
    #entry: [a[0], a[1],x_match, y_match, cor_val]
    #entry alt: [x,x_match, cor_val, subpix, y]
    if(entry_val[2] < 0 or entry_val[3] < 0 or entry_val[4] < threshold):
        return pos_remove,remove_flag,entry_flag
    for i in range(len(res_list)):       
        
        if(res_list[i][0] == entry_val[0] and res_list[i][1] == entry_val[1] and res_list[i][2] == entry_val[2] and res_list[i][3] == entry_val[3] and res_list[i][4] - entry_val[4] < float_epsilon):
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
def cor_pix_norect(Gi,n, xLim, yLim, maskR, xOffset1, xOffset2,yOffset1, yOffset2):
    max_cor = 0.0
    max_index_x = -1
    max_index_y = -1
    agi = np.sum(Gi)/n
    val_i = np.sum((Gi-agi)**2)
    for xi in range(xOffset1, xLim-xOffset2):
        for yi in range(yOffset1, yLim-yOffset2):
            Gt = maskR[:,yi,xi]
            if(np.sum(Gt) != 0): #ignore fully black points
                agt = np.sum(Gt)/n        
                val_t = np.sum((Gt-agt)**2)
                if(val_i > float_epsilon and val_t > float_epsilon): 
                    cor = np.sum((Gi-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))              
                    if cor > max_cor:
                        max_cor = cor
                        max_index_x = xi
                        max_index_y = yi
    return max_index_x,max_index_y,max_cor
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
    start = time.time()
    for i in range(0,imshape[0],100):
        for j in range(0,imshape[1],100):
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
    res_y = []
    for a in pointsL:
        if(np.sum(a) != 0): #dont match fully dark slices
            x_match,y_match,cor_val = cor_pix_norect(a,n,yLim, xLim, maskR, xOffsetL, xOffsetR, yOffsetT, yOffsetB)
                
            pos_remove, remove_flag, entry_flag = compare_cor_fmat(res_y,[a[0], a[1],x_match, y_match, cor_val], thresh)
            if(remove_flag):
                res_y.pop(pos_remove)
                res_y.append([a[0], a[1],x_match, y_match, cor_val])
            elif(entry_flag):
                res_y.append([a[0], a[1],x_match, y_match, cor_val])
    
    #use the resulting point matches to compute the fundamental matrix
    pts1 = []
    pts2 = []
    for i in res_y:
        pts1.append(np.asarray([i[0],i[1]]))
        pts2.append(np.asarray([i[2],i[3]]))
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    end = time.time()
    print("Seconds:" + str(end - start))
    print(F)
    start = time.time()
    F2 = scr.find_f_mat(imgL[0], imgR[0])
    end = time.time()
    print("Seconds:" + str(end - start))
    print(F2)
    recA1,recA2, H1,H2 = scr.rectify_pair(imgL[0], imgR[0],F)
    recB1,recB2, H3,H4 = scr.rectify_pair(imgL[0], imgR[0],F2)
    scr.display_stereo(imgL[0],imgR[0])
    scr.display_stereo(recA1, recA2)
    scr.display_stereo(recB1, recB2)
    
  
    
