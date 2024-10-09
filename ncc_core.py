# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 11:23:50 2023

@author: myuey
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
def cor_acc_rbf(Gi,y,n, xLim, maskR, xOffset1, xOffset2, interp_num):
    '''
    NCC point correlation function with rbf linear interpolation in 8-neighbors of points found

    Parameters
    ----------
    Gi : Numpy array
        Vector with grayscale values of pixel stack to match with
    y : integer
        y position of row of interest
    n : integer
        number of images in image stack
    xLim : integer
        Maximum number for x-dimension of images
    maskR : 2D image stack
        vertical stack of 2D numpy array image data
    xOffset1 : integer
        Offset from left side of image stack to start looking from
    xOffset2 : integer
        Offset from right side of image stack to stop looking at
    interp_num : integer
        Number of subpixel interpolations to make between pixels

    Returns
    -------
    max_index : integer
        identified best matching x coordinate
    max_cor : float
        correlation value of best matching coordinate
    max_mod : list of floats wth 2 entries
        subpixel interpolation coordinates from found matching coordinate

    '''
    #calculate size of interpolation grid
    grid_num = interp_num*2 + 3
    max_cor = 0
    max_index = -1
    max_mod = np.asarray([0.0,0.0]) #default to no change
    agi = np.sum(Gi)/n
    val_i = np.sum((Gi-agi)**2)
    #Search the entire line    
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = maskR[:,y,xi]
        agt = np.sum(Gt)/n        
        val_t = np.sum((Gt-agt)**2)
        if(val_i > float_epsilon and val_t > float_epsilon): 
            cor = np.sum((Gi-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))              
            if cor > max_cor:
                max_cor = cor
                max_index = xi
    y_flag = 0.0
    #search above and below
    Gup = maskR[:,y-1, max_index]
    agup = np.sum(Gup)/n
    val_up = np.sum((Gup-agup)**2)
    
    if(val_i > float_epsilon and val_up > float_epsilon): 
        cor = np.sum((Gi-agi)*(Gup - agup))/(np.sqrt(val_i*val_up))              
        if cor > max_cor:
           max_cor = cor
           max_mod = np.asarray([-1.0,0.0])
           y -= 1
           y_flag = -1.0
    Gdn = maskR[:,y+1, max_index]
    agdn = np.sum(Gdn)/n
    val_dn = np.sum((Gdn-agdn)**2)
    if(val_i > float_epsilon and val_dn > float_epsilon): 
        cor = np.sum((Gi-agi)*(Gdn - agdn))/(np.sqrt(val_i*val_dn))              
        if cor > max_cor:
            max_cor = cor
            max_mod = np.asarray([1.0,0.0]) 
            y+=1
            y_flag = 1.0
    max_mod = np.asarray([y_flag,0.0])
    #search around the found best index
    if(max_index > -1):
        
        
        #define points
        #[C,N,S,W,E,NW,SE,NE,SW]
        mod_neighbor = [(0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
        x_val = []
        y_val = []
        for i in mod_neighbor:
            x_val.append(i[1])
            y_val.append(i[0])
        x_val = np.asarray(x_val)
        y_val = np.asarray(y_val)
        xin = np.linspace(np.min(x_val), np.max(x_val), grid_num)
        yin = np.linspace(np.min(y_val), np.max(y_val), grid_num)
        
        g_len = xin.shape[0]
        h_len = yin.shape[0]
        resG = []
        for u in range(h_len):
            resG.append(xin)
        resHT = []
        for a in yin:
            resHT_R = []
            for b in range(g_len):
               resHT_R.append(a) 
            resHT.append(resHT_R)
            
        resFlatG = []
        for i in resG:
            for j in i:
                resFlatG.append(j)
        resFlatH = []
        for i in resHT:
            for j in i:
                resFlatH.append(j)
        
        xin = np.array(resFlatG)
        yin = np.array(resFlatH)
        
       
        
        z_val_list = [maskR[:,y,max_index],maskR[:,y-1,max_index],maskR[:,y+1,max_index],maskR[:,y,max_index-1],
                        maskR[:,y,max_index+1], maskR[:,y-1,max_index-1],maskR[:,y+1,max_index+1],
                        maskR[:,y-1,max_index+1],maskR[:,y+1,max_index-1]]
        z_val = np.empty((len(z_val_list),len(z_val_list[0])))
        for a in range(len(z_val_list)):
            for b in range(len(z_val_list[0])):
                z_val[a][b] = z_val_list[a][b]
        #Check ncc values for known neighboring points
        for a in range(1,len(z_val)):
            Gt = z_val[a]
            agt = np.sum(Gt)/n        
            val_t = np.sum((Gt-agt)**2)
            if(val_i > float_epsilon and val_t > float_epsilon): 
                cor = np.sum((Gi-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))              
                if cor > max_cor:
                    max_cor = cor
                    max_mod += np.asarray([mod_neighbor[a][0], mod_neighbor[a][1]])
                    
        #create interpolation fields with specified resolution and add to list  
        interp_fields_list = []
        for s in range(z_val.shape[1]):
            obs = np.vstack((x_val, y_val)).T
            interp = np.vstack((xin, yin)).T
            d0=np.empty((obs[:,0].shape[0],interp[:,0].shape[0]))
            for i in numba.prange(obs[:,0].shape[0]):
                for j in range(interp[:,0].shape[0]):
                    d0[i][j] = obs[:,0][i]-interp[:,0][j]
            
            d1=np.empty((obs[:,1].shape[0],interp[:,1].shape[0]))
            for i in numba.prange(obs[:,1].shape[0]):
                for j in range(interp[:,1].shape[0]):
                    d1[i][j]=obs[:,1][i]-interp[:,1][j]
            dist = np.hypot(d0, d1)
            
            interp0 = np.vstack((x_val, y_val)).T
            d0=np.empty((obs[:,0].shape[0],interp0[:,0].shape[0]))
            for i in numba.prange(obs[:,0].shape[0]):
                for j in range(interp0[:,0].shape[0]):
                    d0[i][j] = obs[:,0][i]-interp0[:,0][j]
            
            d1=np.empty((obs[:,1].shape[0],interp0[:,1].shape[0]))
            for i in numba.prange(obs[:,1].shape[0]):
                for j in range(interp0[:,1].shape[0]):
                    d1[i][j]=obs[:,1][i]-interp0[:,1][j]
            internal_dist = np.hypot(d0, d1)
            
            weights = np.linalg.solve(internal_dist, z_val[:,s])
            zi =  np.dot(dist.T, weights)
            grid = zi.reshape((grid_num, grid_num))
            interp_fields_list.append(grid)
        #calculate increments of interpolation field coordinates
        dist_inc = 1/interp_num 
        interp_fields = np.empty((len(interp_fields_list),len(interp_fields_list[0]),len(interp_fields_list[0][0])))
        for a in range(len(interp_fields_list)):
            for b in range(len(interp_fields_list[0])):
                for c in range(len(interp_fields_list[0][0])):
                    interp_fields[a][b][c] = interp_fields_list[a][b][c]
        #Pull pixel stacks from interpolation field stack and check with ncc  
        for i in range(interp_fields.shape[1]):
            for j in range(interp_fields.shape[2]):

                if not j*dist_inc % 1 == 0 and  not i*dist_inc % 1 == 0 :

                    Gt = interp_fields[:,i,j]
                    agt = np.sum(Gt)/n        
                    val_t = np.sum((Gt-agt)**2)
                    if(val_i > float_epsilon and val_t > float_epsilon): 
                        cor = np.sum((Gi-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))              
                        if cor > max_cor:
                            max_cor = cor
                            max_mod += np.asarray([j*dist_inc, i*dist_inc])
    return max_index,max_cor,max_mod

@numba.jit(nopython=True)
def cor_acc_linear(Gi,y,n, xLim, maskR, xOffset1, xOffset2, interp_num):
    '''
    NCC point correlation function with simple linear interpolation in 8-neighbors of points found

    Parameters
    ----------
    Gi : Numpy array
        Vector with grayscale values of pixel stack to match with
    y : integer
        y position of row of interest
    n : integer
        number of images in image stack
    xLim : integer
        Maximum number for x-dimension of images
    maskR : 2D image stack
        vertical stack of 2D numpy array image data
    xOffset1 : integer
        Offset from left side of image stack to start looking from
    xOffset2 : integer
        Offset from right side of image stack to stop looking at
    interp_num : integer
        Number of subpixel interpolations to make between pixels

    Returns
    -------
    max_index : integer
        identified best matching x coordinate
    max_cor : float
        correlation value of best matching coordinate
    max_mod : list of floats wth 2 entries
        subpixel interpolation coordinates from found matching coordinate

    '''
    max_cor = 0
    max_index = -1
    max_mod = np.asarray([0.0,0.0]) #default to no change
    agi = np.sum(Gi)/n
    val_i = np.sum((Gi-agi)**2)
    #Search the entire line    
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = maskR[:,y,xi]
        agt = np.sum(Gt)/n        
        val_t = np.sum((Gt-agt)**2)
        if(val_i > float_epsilon and val_t > float_epsilon): 
            cor = np.sum((Gi-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))              
            if cor > max_cor:
                max_cor = cor
                max_index = xi
    y_flag = 0.0
    #search above and below
    Gup = maskR[:,y-1, max_index]
    agup = np.sum(Gup)/n
    val_up = np.sum((Gup-agup)**2)
    
    if(val_i > float_epsilon and val_up > float_epsilon): 
        cor = np.sum((Gi-agi)*(Gup - agup))/(np.sqrt(val_i*val_up))              
        if cor > max_cor:
           max_cor = cor
           max_mod = np.asarray([-1.0,0.0])
           y -= 1
           y_flag = -1.0
    Gdn = maskR[:,y+1, max_index]
    agdn = np.sum(Gdn)/n
    val_dn = np.sum((Gdn-agdn)**2)
    if(val_i > float_epsilon and val_dn > float_epsilon): 
        cor = np.sum((Gi-agi)*(Gdn - agdn))/(np.sqrt(val_i*val_dn))              
        if cor > max_cor:
            max_cor = cor
            max_mod = np.asarray([1.0,0.0]) 
            y+=1
            y_flag = 1.0
    #search around the found best index
    if(max_index > -1):  
        increment = 1/ (interp_num + 1)
        
        #define points
        G_cent =  maskR[:,y,max_index]
        #[N,S,W,E]
        coord_card = [(-1,0),(1,0),(0,-1),(0,1)]
        #[NW,SE,NE,SW]
        coord_diag = [(-1,-1),(1,1),(-1,1),(1,-1)]
        G_card = [maskR[:,y-1,max_index],maskR[:,y+1,max_index],maskR[:,y,max_index-1],
                          maskR[:,y,max_index+1]]
        G_diag = [maskR[:,y-1,max_index-1],maskR[:,y+1,max_index+1],maskR[:,y-1,max_index+1],
                          maskR[:,y+1,max_index-1]]
        #define loops
        #check cardinal
        for i in range(len(coord_card)):
            val = G_card[i] - G_cent
            if(i<2):#Redundant to run ncc on EW to check for better value due to line-search covering them eventually
                G_check = G_card[i]
                ag_check = np.sum(G_check)/n
                val_check = np.sum((G_check-ag_check)**2)
                if(val_i > float_epsilon and val_check > float_epsilon): 
                    cor = np.sum((Gi-agi)*(G_check - ag_check))/(np.sqrt(val_i*val_check))     
                    if cor > max_cor:
                        max_cor = cor
                        max_mod = max_mod+np.asarray([coord_card[i][0]*1.0, coord_card[i][1]*1.0])
            for j in range(interp_num):
                G_check= ((j+1)*increment * val) + G_cent
                ag_check = np.sum(G_check)/n
                val_check = np.sum((G_check-ag_check)**2)
                if(val_i > float_epsilon and val_check > float_epsilon): 
                    cor = np.sum((Gi-agi)*(G_check - ag_check))/(np.sqrt(val_i*val_check))
                     
                    if cor > max_cor:
                        max_cor = cor
                        max_mod = max_mod+np.asarray([coord_card[i][0]*(j+1)*increment,coord_card[i][1]*(j+1)*increment])
                        
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
                        max_mod = max_mod+np.asarray([coord_diag[i][0]*(j+1)*increment,coord_diag[i][1]*(j+1)*increment])      
    max_mod = max_mod+np.asarray([y_flag,0.0])
    return max_index,max_cor,max_mod
@numba.jit(nopython=True)
def cor_acc_lin2(Gi,y,n, xLim, maskR, xOffset1, xOffset2, interp_num):
    max_cor = 0
    max_index = -1
    max_mod = np.asarray([0.0,0.0]) #default to no change
    agi = np.sum(Gi)/n
    val_i = np.sum((Gi-agi)**2)
    #Search the entire line    
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = maskR[:,y,xi]
        agt = np.sum(Gt)/n        
        val_t = np.sum((Gt-agt)**2)
        if(val_i > float_epsilon and val_t > float_epsilon): 
            cor = np.sum((Gi-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))              
            if cor > max_cor:
                max_cor = cor
                max_index = xi
    y_flag = 0.0
    #search above and below
    Gup = maskR[:,y-1, max_index]
    agup = np.sum(Gup)/n
    val_up = np.sum((Gup-agup)**2)
    
    if(val_i > float_epsilon and val_up > float_epsilon): 
        cor = np.sum((Gi-agi)*(Gup - agup))/(np.sqrt(val_i*val_up))              
        if cor > max_cor:
           max_cor = cor
           max_mod = np.asarray([-1.0,0.0])
           y -= 1
           y_flag = -1.0
    
    Gup2 = maskR[:,y-2, max_index]
    agup2 = np.sum(Gup2)/n
    val_up2 = np.sum((Gup2-agup2)**2)
    
    if(val_i > float_epsilon and val_up2 >  float_epsilon): 
        cor = np.sum((Gi-agi)*(Gup2 - agup2))/(np.sqrt(val_i*val_up2))              
        if cor > max_cor:
           max_cor = cor
           max_mod = np.asarray([-2.0,0.0])
           y -= 2
           y_flag = -2.0       
    
    Gdn = maskR[:,y+1, max_index]
    agdn = np.sum(Gdn)/n
    val_dn = np.sum((Gdn-agdn)**2)
    if(val_i > float_epsilon and val_dn > float_epsilon): 
        cor = np.sum((Gi-agi)*(Gdn - agdn))/(np.sqrt(val_i*val_dn))              
        if cor > max_cor:
            max_cor = cor
            max_mod = np.asarray([1.0,0.0]) 
            y+=1
            y_flag = 1.0
    
    Gdn2 = maskR[:,y+2, max_index]
    agdn2 = np.sum(Gdn2)/n
    val_dn2 = np.sum((Gdn2-agdn2)**2)
    if(val_i > float_epsilon and val_dn2 > float_epsilon): 
        cor = np.sum((Gi-agi)*(Gdn2 - agdn2))/(np.sqrt(val_i*val_dn2))              
        if cor > max_cor:
            max_cor = cor
            max_mod = np.asarray([2.0,0.0]) 
            y+=2
            y_flag = 2.0
            
    if(max_index > -1):  
        increment = 1/ (interp_num + 1)
        
        #define points
        G_cent =  maskR[:,y,max_index]
        #[N,S,W,E]
        coord_card = [(-1,0),(1,0),(0,-1),(0,1)]
        #[NW,SE,NE,SW]
        coord_diag = [(-1,-1),(1,1),(-1,1),(1,-1)]
        
        G_card = [maskR[:,y-1,max_index],maskR[:,y+1,max_index],maskR[:,y,max_index-1],
                          maskR[:,y,max_index+1]]
        G_diag = [maskR[:,y-1,max_index-1],maskR[:,y+1,max_index+1],maskR[:,y-1,max_index+1],
                          maskR[:,y+1,max_index-1]]

        #define loops
        #check cardinal
        
        for i in range(len(coord_card)):
            val = G_card[i] - G_cent
            if(i<2):#Redundant to run ncc on EW to check for better value due to line-search covering them eventually
                G_check = G_card[i]
                ag_check = np.sum(G_check)/n
                val_check = np.sum((G_check-ag_check)**2)
                if(val_i > float_epsilon and val_check > float_epsilon): 
                    cor = np.sum((Gi-agi)*(G_check - ag_check))/(np.sqrt(val_i*val_check))     
                    if cor > max_cor:
                        max_cor = cor
                        max_mod = max_mod+np.asarray([coord_card[i][0]*1.0, coord_card[i][1]*1.0])
            for j in range(interp_num):
                G_check= ((j+1)*increment * val) + G_cent
                ag_check = np.sum(G_check)/n
                val_check = np.sum((G_check-ag_check)**2)
                if(val_i > float_epsilon and val_check > float_epsilon): 
                    cor = np.sum((Gi-agi)*(G_check - ag_check))/(np.sqrt(val_i*val_check))
                     
                    if cor > max_cor:
                        max_cor = cor
                        max_mod = max_mod+np.asarray([coord_card[i][0]*(j+1)*increment,coord_card[i][1]*(j+1)*increment])
                        
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
                        max_mod = max_mod+np.asarray([coord_diag[i][0]*(j+1)*increment,coord_diag[i][1]*(j+1)*increment])      
    max_mod = max_mod+np.asarray([y_flag,0.0])
    return max_index,max_cor,max_mod

@numba.jit(nopython=True)
def cor_acc_pix(Gi,y,n, xLim, maskR, xOffset1, xOffset2):
    '''
    NCC point correlation function with no subpixel interpolation

    Parameters
    ----------
    Gi : Numpy array
        Vector with grayscale values of pixel stack to match with
    y : integer
        y position of row of interest
    n : integer
        number of images in image stack
    xLim : integer
        Maximum number for x-dimension of images
    maskR : 2D image stack
        vertical stack of 2D numpy array image data
    xOffset1 : integer
        Offset from left side of image stack to start looking from
    xOffset2 : integer
        Offset from right side of image stack to stop looking at

    Returns
    -------
    max_index : integer
        identified best matching x coordinate
    max_cor : float
        correlation value of best matching coordinate
    max_mod : list of floats wth 2 entries
        modifier to apply to best matching coordinate if the actual best is above or below.

    '''
    max_cor = 0.0
    max_index = -1
    max_mod = [0,0] #default to no change
    agi = np.sum(Gi)/n
    val_i = np.sum((Gi-agi)**2)
    #Search the entire line    
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = maskR[:,y,xi]
        agt = np.sum(Gt)/n        
        val_t = np.sum((Gt-agt)**2)
        if(val_i > float_epsilon and val_t > float_epsilon): 
            cor = np.sum((Gi-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))              
            if cor > max_cor:
                max_cor = cor
                max_index = xi
    #search surroundings of found best match
    Gup = maskR[:,y-1, max_index]
    agup = np.sum(Gup)/n
    val_up = np.sum((Gup-agup)**2)
    if(val_i > float_epsilon and val_up > float_epsilon): 
        cor = np.sum((Gi-agi)*(Gup - agup))/(np.sqrt(val_i*val_up))              
        if cor > max_cor:
           max_cor = cor
           max_mod = [-1,0]
    
    Gdn = maskR[:,y+1, max_index]
    agdn = np.sum(Gdn)/n
    val_dn = np.sum((Gdn-agdn)**2)
    if(val_i > float_epsilon and val_dn > float_epsilon): 
        cor = np.sum((Gi-agi)*(Gdn - agdn))/(np.sqrt(val_i*val_dn))              
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
def cor_pts(config):
    '''
    Runs correlation functions only

    Parameters
    ----------
    config : confighandler
        Object storing parameters for the function

    Returns
    -------
    ptsL : list of 2d lists
        DESCRIPTION.
    ptsR : list of 2d lists
        DESCRIPTION.

    '''
    kL, kR, r_vec, t_vec, F, imgL, imgR, imshape, maskL, maskR, col_refL, col_refR = startup_load(config)
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
    for y in tqdm(range(yOffsetT, yLim-yOffsetB)):
        res_y = []
        for x in range(xOffsetL, xLim-xOffsetR, interval):
            Gi = maskL[:,y,x]
            if(np.sum(Gi) != 0): #dont match fully dark slices
                if config.speed_mode:
                    x_match,cor_val,subpix = cor_acc_pix(Gi,y,n, xLim, maskR, xOffsetL, xOffsetR)
                else:    
                    x_match,cor_val,subpix = cor_acc_rbf(Gi,y,n, xLim, maskR, xOffsetL, xOffsetR, interp)
                    
                pos_remove, remove_flag, entry_flag = compare_cor(res_y,
                                                                  [x,x_match, cor_val, subpix, y], thresh)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, subpix, y])
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, subpix, y])
        rect_res.append(res_y)
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




    return ptsL,ptsR

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
            Gi = maskL[:,y,x]
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                if config.speed_mode:
                    x_match,cor_val,subpix = cor_acc_pix(Gi,y,n, xLim, maskR, xOffsetL, xOffsetR)
                else:
                    if config.interp_mode == 0:

                        x_match,cor_val,subpix = cor_acc_linear(Gi,y,n, xLim, maskR, xOffsetL, xOffsetR, interp)
                    else:

                        x_match,cor_val,subpix = cor_acc_rbf(Gi,y,n, xLim, maskR, xOffsetL, xOffsetR, interp)

                
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
            if config.col_first:
                col_arr = scr.gen_color_arr_black(len(ptsL))
            else:
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

