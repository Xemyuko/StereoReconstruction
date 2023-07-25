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
float_epsilon = 1e-9
def startup_load(config):
    print("Loading files...")
    kL,kR,r_vec,t_vec = scr.initial_load(config.tmod, config.mat_folder, config.kL_file, 
                                         config.kR_file, config.R_file, config.t_file, 
                                         config.skiprow,config.delim)
    kL_inv = np.linalg.inv(kL)
    kR_inv = np.linalg.inv(kR)
    #Load images
    imgL,imgR = scr.load_images(folderL = config.left_folder, folderR = config.right_folder)
    imshape = imgL[0].shape
    #rectify images
    fund_mat = None
    if os.path.isfile(config.mat_folder + config.f_file) and config.f_load == 1:
        fund_mat = np.loadtxt(config.mat_folder + config.f_file, skiprows=config.skiprow, delimiter = config.delim)
        print("Fundamental Matrix Loaded From File: " + config.mat_folder + config.f_file)
    else:
        F = scr.find_f_mat(imgL[0],imgR[0])
        if config.f_save == 1:
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
    #define constants for window
    xLim = imshape[1]
    yLim = imshape[0]
    return kL, kR, r_vec, t_vec, kL_inv, kR_inv, fund_mat, imgL, imgR, imshape, maskL, maskR, xLim, yLim

@numba.jit(nopython=True)
def cor_acc_linear(Gi,x,y,n, xLim, maskR, xOffset1, xOffset2, interp_num):
    max_cor = 0
    max_index = -1
    max_mod = [0.0,0.0] #default to no change
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
            if(i<2):
                G_check = G_card[i]
                ag_check = np.sum(G_check)/n
                val_check = np.sum((G_check-ag_check)**2)
                if(val_i > float_epsilon and val_check > float_epsilon): 
                    cor = np.sum((Gi-agi)*(G_check - ag_check))/(np.sqrt(val_i*val_check))     
                    if cor > max_cor:
                        max_cor = cor
                        max_mod = [coord_card[i][0]*1.0, coord_card[i][1]*1.0]
            for j in range(interp_num):
                G_check= ((j+1)*increment * val) + G_cent
                ag_check = np.sum(G_check)/n
                val_check = np.sum((G_check-ag_check)**2)
                if(val_i > float_epsilon and val_check > float_epsilon): 
                    cor = np.sum((Gi-agi)*(G_check - ag_check))/(np.sqrt(val_i*val_check))
                     
                    if cor > max_cor:
                        max_cor = cor
                        max_mod = [coord_card[i][0]*(j+1)*increment,coord_card[i][1]*(j+1)*increment]
                        
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
                        max_mod = [coord_diag[i][0]*(j+1)*increment,coord_diag[i][1]*(j+1)*increment]      
    return max_index,max_cor,max_mod

@numba.jit(nopython=True)
def cor_acc_pix(Gi,x,y,n, xLim, maskR, xOffset1, xOffset2):
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
                    
#duplicate comparison and correlation thresholding, run when trying to add points to results
def compare_cor(res_list, entry_val, threshold, recon = True):
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
    
def run_cor(config, mapgen = False):
    
    kL, kR, r_vec, t_vec, kL_inv, kR_inv, F, imgL, imgR, imshape, maskL, maskR, xLim, yLim = startup_load(config)
    xOffsetL = config.x_offset_L
    xOffsetR = config.x_offset_R
    yOffsetT = config.y_offset_T
    yOffsetB = config.y_offset_B
    thresh = config.thresh
    interp = config.interp
    rect_res = []
    n = len(imgL)
    interval = 1
    if config.speed_mode > 0:
        interval = config.speed_interval
        print("Speed Mode is on. Correlation results will use an interval spacing of " + str(interval) + 
              " between every pixel checked and no subpixel interpolation will be used.")
    print("Correlating Points...")
    for y in tqdm(range(yOffsetT, yLim-yOffsetB)):
        res_y = []
        for x in range(xOffsetL, xLim-xOffsetR, interval):
            Gi = maskL[:,y,x]
            if(np.sum(Gi) != 0): #dont match fully dark slices
                if config.speed_mode > 0:
                    x_match,cor_val,subpix = cor_acc_pix(Gi,x,y,n, xLim, maskR, xOffsetL, xOffsetR)
                else:    
                    x_match,cor_val,subpix = cor_acc_linear(Gi,x,y,n, xLim, maskR, xOffsetL, xOffsetR, interp)
                    
                pos_remove, remove_flag, entry_flag = compare_cor(res_y,
                                                                  [x,x_match, cor_val, subpix], thresh)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, subpix])
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, subpix])
        rect_res.append(res_y)
        
    if(mapgen):
        res_map = np.zeros((maskL.shape[1],maskL.shape[2]), dtype = 'uint8')
        for i in range(len(rect_res)):
            b = rect_res[i]
            for j in b:
                res_map[i+yOffsetT,j[0]] = j[2]*255
        scr.write_img(res_map, config.corr_map_name)
        print("Correlation Map Creation Complete.")
    
    else:  
        if config.corr_map_out > 0:
            res_map = np.zeros((maskL.shape[1],maskL.shape[2]), dtype = 'uint8')
            for i in range(len(rect_res)):
                b = rect_res[i]
                for j in b:
                    res_map[i+yOffsetT,j[0]] = j[2]*255
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
                sL = HL[2,0]*q[0] + HL[2,1] * (a+yOffsetT) + HL[2,2]
                pL = hL_inv @ np.asarray([[q[0]],[a+yOffsetT],[sL]])
                sR = HR[2,0]*(q[1] + q[3][1]) + HR[2,1] * (a+yOffsetT+q[3][0]) + HR[2,2]
                pR = hR_inv @ np.asarray([[q[1]+ q[3][1]],[a+yOffsetT+q[3][0]],[sR]])
                ptsL.append([pL[0,0],pL[1,0],pL[2,0]])
                ptsR.append([pR[0,0],pR[1,0],pR[2,0]])


        #Triangulate 3D positions from point lists
        #take 2D
        ptsL = scr.conv_pts(ptsL)
        ptsR = scr.conv_pts(ptsR)
        col_arr = scr.gen_color_arr_black(len(ptsL))
        print("Triangulating Points...")
        tri_res = scr.triangulate_list(ptsL,ptsR, r_vec, t_vec, kL_inv, kR_inv, config.precise)
        #Convert numpy arrays to ply point cloud file
        scr.convert_np_ply(np.asarray(tri_res), col_arr,config.output)
        if(config.data_out > 0):
            cor = []
            for i in range(len(rect_res)):
                b = rect_res[i]
                for j in b:
                    cor.append(j[2])
            scr.create_xyz(ptsL,ptsR,cor,tri_res,col_arr, config.data_name)
        print("Reconstruction Complete.")