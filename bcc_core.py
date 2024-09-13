# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:22:04 2024

@author: M
"""

import numpy as np
import scripts as scr
import numba
import os
from tqdm import tqdm
import cv2
float_epsilon = 1e-9



        
def startup_load(config):
    '''
    Loads inputs from config file. Also applies rectification and initial filters.    
    
    Parameters
    ----------
    config : confighandler object
        Configuration file loader

    Returns
    -------
    kL : 3x3 numpy array
        Left camera matrix
    kR : 3x3 numpy array
        Right camera matrix
    r_vec : 3x3 numpy array
        Rotation matrix
    t_vec : numpy array
        Translation vector
    kL_inv : 3x3 numpy array
        Inverse of left camera matrix
    kR_inv : 3x3 numpy array
        Inverse of right camera matrix
    fund_mat : 3x3 numpy array
        Fundamental matrix
    imgL : numpy array
        Input left camera images
    imgR : numpy array
        Input right camera images
    imshape : tuple
        shape of image inputs
    maskL : numpy array
        masked and filtered left images
    maskR : numpy array
        masked and filtered right images

    '''
    print("Loading files...")
    kL,kR,r_vec,t_vec = scr.load_mats(config.mat_folder, config.kL_file, 
                                         config.kR_file, config.R_file, config.t_file, 
                                         config.skiprow,config.delim)
    #Load images
    if(config.sing_img_mode):
        imgL,imgR = scr.load_images_1_dir(config.sing_img_folder, config.sing_left_ind, config.sing_right_ind, config.sing_ext)
    else:
        imgL,imgR = scr.load_images(folderL = config.left_folder, folderR = config.right_folder)
    imshape = imgL[0].shape
    
    #undistort images if set for
    if config.distort_comp:
        #load distortion vectors
        dL = np.loadtxt(config.mat_folder +config.left_distort, skiprows=config.skiprow, delimiter = config.delim)
        dR = np.loadtxt(config.mat_folder +config.right_distort, skiprows=config.skiprow, delimiter = config.delim)
        #undistort images and update camera matrices
        kL,imgL= scr.undistort(imgL, kL,dL)
        kR,imgR= scr.undistort(imgR, kR,dR)
   
    #rectify images
    fund_mat = None
    if os.path.isfile(config.mat_folder + config.f_file) and config.f_mat_file_mode == 1:
        fund_mat = np.loadtxt(config.mat_folder + config.f_file, skiprows=config.skiprow, delimiter = config.delim)
        print("Fundamental Matrix Loaded From File: " + config.mat_folder + config.f_file)
    else:
        F=None
        if config.f_search:
            F = scr.find_f_mat_list(imgL,imgR, config.f_mat_thresh, config.f_calc_mode)
        else:
            if(config.f_mat_ncc):
                F = scr.find_f_mat_ncc(imgL,imgR,config.f_mat_thresh, config.f_calc_mode)
            else:
                
                F = scr.find_f_mat(imgL[0],imgR[0], config.f_mat_thresh, config.f_calc_mode)
        if config.f_mat_file_mode == 2:
            print("Fundamental Matrix Saved To File: " + config.mat_folder + config.f_file)
            np.savetxt(config.mat_folder + config.f_file, F)
            with open(config.mat_folder + config.f_file, 'r') as ori:
                oricon = ori.read()
            with open(config.mat_folder + config.f_file, 'w') as ori:  
                ori.write("3\n3\n")
                ori.write(oricon)
        fund_mat = F
    rectL,rectR = scr.rectify_lists(imgL,imgR, fund_mat)
    avgL = np.asarray(rectL).mean(axis=(0))
    avgR = np.asarray(rectR).mean(axis=(0))

    #Background filter
    thresh_val = config.mask_thresh
    maskL = scr.mask_avg_list(avgL,rectL, thresh_val)
    maskR = scr.mask_avg_list(avgR,rectR, thresh_val)

    maskL = np.asarray(maskL)
    maskR = np.asarray(maskR)
    col_refL = None
    col_refR = None
    if config.color_recon:
        col_refL, col_refR= scr.load_images_1_dir(config.sing_img_folder, config.sing_left_ind, config.sing_right_ind, config.sing_ext, colorIm = True)
    
    return kL, kR, r_vec, t_vec, fund_mat, imgL, imgR, imshape, maskL, maskR, col_refL, col_refR
@numba.jit(nopython=True)
def cor_bin_pix(Gi,y,n, xLim, maskR, xOffset1, xOffset2):
    max_cor = 0.0
    max_index = -1
    max_mod = [0,0] #default to no change
    
    
    #Search the entire line    
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = maskR[:,y,xi].astype('int16')
        vc = np.sum((Gi-Gt)**2)/n
        if vc > 0:
            cor = 1-vc
        else:
            cor = 1+vc
        if cor > max_cor:
            max_cor = cor
            max_index = xi
    #search surroundings of found best match
    Gup = maskR[:,y-1, max_index].astype('int16')
    vc = np.sum((Gi-Gup)**2)/n
    if vc > 0:
        cor = 1-vc
    else:
        cor = 1+vc    
    if cor > max_cor:
        max_cor = cor
        max_mod = [-1,0]
    
    Gdn = maskR[:,y+1, max_index].astype('int16')
    vc = np.sum((Gi-Gdn)**2)/n
    if vc > 0:
        cor = 1-vc
    else:
        cor = 1+vc           
    if cor > max_cor:
        max_cor = cor
        max_mod = [1,0]        
    return max_index,max_cor,max_mod
def compare_cor(res_list, entry_val, threshold, recon = True):
    '''
    Checks proposed additions to the list of correlated points for duplicates, threshold requirements, and existing matches
    
    
    Parameters
    ----------
    res_list : list of entries
        Existing list of entries
    entry_val : list of values in the format: [x,x_match, cor_val, subpix, y]
        x : Left image x value of pixel stack
        x_match: Matched right image pixel stack x value
        cor_val: Correlation score for match
        subpix: Subpixel interpolation coordinates
        y: y value of the rectified line that x and x-match are found in
    threshold : float
        Minimum correlation value needed to be added to list of results
    recon : boolean, optional
        Controls if the threshold needs to be met, which is not needed for making a correlation map. The default is True.

    Returns
    -------
    pos_remove : integer
        position of entry to remove from list
    remove_flag : boolean
        True if an entry needs to be removed
    entry_flag : boolean
        True if entry_val is a valid addition to the result list

    '''
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
def create_diff_grid(img_stk, thresh = 10):
    res_grid_stk = []
    for i in range(len(img_stk) - 1):
        comp1 = img_stk[i]
        comp2 = img_stk[i+1]
        res_grid_stk.append(comp1-comp2)
    for i in res_grid_stk:
        i[i<-thresh] = 0
        i[i>thresh] = 0
        i[i!=0] = 1
    res_grid_stk = np.asarray(res_grid_stk)
    return res_grid_stk

def create_diff_grid2(img_stk, thresh_close = 10):
    res_grid_stk = []
    thresh_mean = np.mean(img_stk)
    
    for i in range(len(img_stk) - 1):
        comp1 = img_stk[i]
        comp2 = img_stk[i+1]
        res_grid_stk.append(comp1-comp2)
        
    for a in res_grid_stk:
        a[a<-thresh_close] = 0
        a[a>thresh_close] = 0
        a[a!=0] = 1
    
    for b in img_stk:
        b[b<=thresh_mean] = 0
        b[b>thresh_mean] = 1
        res_grid_stk.append(b)
    
    res_grid_stk = np.asarray(res_grid_stk)
    return res_grid_stk
def run_cor(config, mapgen = False):
    '''
    Primary function, runs correlation and triangulation functions, then creates a point cloud .ply file of the results. 

    Parameters
    ----------
    config : confighandler
        Object storing parameters for the function
    mapgen : Boolean, optional
        Controls if the function will also create a correlation map image file. The default is False.

    Returns
    -------
    None.

    '''
    kL, kR, r_vec, t_vec, F, imgL, imgR, imshape, maskL, maskR,col_refL, col_refR = startup_load(config)
    maskL = create_diff_grid2(maskL)
    maskR = create_diff_grid2(maskR)

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
    if config.speed_mode:
        interval = config.speed_interval
        print("Speed Mode is on. Correlation results will use an interval spacing of " + str(interval) + 
              " between every column checked and no subpixel interpolation will be used.")
    print("Correlating Points...")
    for y in tqdm(range(yOffsetT, yLim-yOffsetB)):
        res_y = []
        for x in range(xOffsetL, xLim-xOffsetR, interval):
            Gi = maskL[:,y,x].astype('int16')
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                x_match,cor_val,subpix = cor_bin_pix(Gi,y,n, xLim, maskR, xOffsetL, xOffsetR)

                pos_remove, remove_flag, entry_flag = compare_cor(res_y,
                                                                  [x,x_match, cor_val, subpix, y], thresh)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, subpix, y])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, subpix, y])
        
        rect_res.append(res_y)
        
    if(mapgen):
        res_map = np.zeros((maskL.shape[1],maskL.shape[2]), dtype = 'uint8')
        for i in range(len(rect_res)):
            b = rect_res[i]
            for j in b:
                res_map[i+yOffsetT,j[0]] = j[2]*255
        color1 = (0,0,255)
        #stack res_map
        res_map = np.stack((res_map,res_map,res_map),axis = 2)
        line_thick = 2
        res_map = cv2.rectangle(res_map, (xOffsetL,yOffsetT), (xLim - xOffsetR,yLim - yOffsetB), color1,line_thick)
        scr.write_img(res_map, config.corr_map_name)
        print("Correlation Map Creation Complete.")
    
    else:  
        if config.corr_map_out:
            res_map = np.zeros((maskL.shape[1],maskL.shape[2]), dtype = 'uint8')
            for i in range(len(rect_res)):
                b = rect_res[i]
                for j in b:
                    res_map[i+yOffsetT,j[0]] = j[2]*255
            color1 = (0,0,255)
            #stack res_map
            res_map = np.stack((res_map,res_map,res_map),axis = 2)
            line_thick = 2
            res_map = cv2.rectangle(res_map, (xOffsetL,yOffsetT), (xLim - xOffsetR,yLim - yOffsetB), color1,line_thick)
            scr.write_img(res_map, config.corr_map_name)
            print("Correlation Map Creation Complete.")
        #Convert matched points from rectified space back to normal space
        im_a,im_b,HL,HR = scr.rectify_pair(imgL[0],imgR[0], F)
        hL_inv = np.linalg.inv(HL)
        hR_inv = np.linalg.inv(HR)
        ptsL = []
        ptsR = []
        for a in range(len(rect_res)):
            b = rect_res[a]
            for q in b:
                xL = q[0]
                y = q[4]
                xR = q[1]
                xL_u = (hL_inv[0,0]*xL + hL_inv[0,1] * y + hL_inv[0,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * y + hL_inv[2,2])
                yL_u = (hL_inv[1,0]*xL + hL_inv[1,1] * y + hL_inv[1,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * y + hL_inv[2,2])
                xR_u = (hR_inv[0,0]*xR + hR_inv[0,1] * y + hR_inv[0,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * y + hR_inv[2,2])
                yR_u = (hR_inv[1,0]*xR + hR_inv[1,1] * y + hR_inv[1,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * y + hR_inv[2,2])
                ptsL.append([xL_u,yL_u])
                ptsR.append([xR_u,yR_u])


        #Triangulate 3D positions from point lists

        col_arr = None
        if config.color_recon:
            col_ptsL = np.around(ptsL,0).astype('uint16')
            col_ptsR = np.around(ptsR,0).astype('uint16')
            col_arr = scr.get_color(col_refL, col_refR, col_ptsL, col_ptsR)
        else:
            col_arr = scr.gen_color_arr_black(len(ptsL))
        print("Triangulating Points...")
        tri_res = scr.triangulate_list(ptsL,ptsR, r_vec, t_vec, kL, kR)
        #Convert numpy arrays to ply point cloud file
        if('.pcf' in config.output):
            cor = []
            for i in range(len(rect_res)):
                b = rect_res[i]
                for j in b:
                    cor.append(j[2])
            scr.create_pcf(ptsL,ptsR,cor,np.asarray(tri_res),col_arr, config.output)
        else:
            scr.convert_np_ply(np.asarray(tri_res), col_arr,config.output)
        if(config.data_out):
            cor = []
            for i in range(len(rect_res)):
                b = rect_res[i]
                for j in b:
                    cor.append(j[2])
            scr.create_data_out(ptsL,ptsR,cor,tri_res,col_arr, config.data_name)
        print("Reconstruction Complete.")
