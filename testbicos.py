# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:55:18 2025

@author: myuey
"""

import numpy as np
import scripts as scr
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numba
import itertools as itt
import scipy.signal as sig
#used for comparing floating point numbers to avoid numerical errors
float_epsilon = 1e-9


def biconv3(imgs, n =8):
    #Add neighbors together, then compare combinations
    imshape = imgs[0].shape

    comN = 4
    imgs1b = np.zeros((n,imshape[0],imshape[1]))
    
    for i in range(0,2*n,2):
        imgs1b[int(i/2),:,:] = imgs[i,:,:] + imgs[i+1,:,:]
   
    combs = list(itt.combinations(range(1, n + 1), comN))
    perm_combs = []

    for comb in combs:
        perm_combs.extend(itt.permutations(comb))

    perm_combs = np.array(sorted(perm_combs))
    # Remove unwanted permutations
    perm_combs = perm_combs[(perm_combs[:, 2] <= perm_combs[:, 3]) &
                         (perm_combs[:, 0] <= perm_combs[:, 1]) &
                         (perm_combs[:, 0] <= perm_combs[:, 2])]         
     
    bilength = perm_combs.shape[0]
    res_stack1 = np.zeros((bilength,imshape[0],imshape[1]),dtype = 'int8')
    for indval in tqdm(range(bilength)):
        i, j, k, l = perm_combs[indval]
        res_stack1[indval,:,:] = (imgs1b[i-1,:,:] + imgs1b[j-1,:,:]) > (imgs1b[k-1,:,:] + imgs1b[l-1,:,:])

    return res_stack1
    

def biconv1(imgs, n = 8):
    #Compare with average
    imshape = imgs[0].shape
    imgs1a = np.zeros((n,imshape[0],imshape[1]))
    imgs1b = np.zeros((n,imshape[0],imshape[1]))
    for a in range(n):
        imgs1a[a,:,:]  = imgs[a,:,:]
        
    avg_img = imgs1a.mean(axis=(0))
    for b in range(n):
        imgs1b[b,:,:] = imgs1a[b,:,:] > avg_img
    return imgs1b
def biconv2(imgs, n = 8):
    comN = 4
    #Compare combinations of images, each image with every other image
    imshape = imgs[0].shape
    imgs1a = np.zeros((n,imshape[0],imshape[1]))

    for a in range(n):
        imgs1a[a,:,:]  = imgs[a,:,:]

    combs = list(itt.combinations(range(1, n + 1), comN))
    perm_combs = []

    for comb in combs:
        perm_combs.extend(itt.permutations(comb))

    perm_combs = np.array(sorted(perm_combs))
   # Remove unwanted permutations
    perm_combs = perm_combs[(perm_combs[:, 2] <= perm_combs[:, 3]) &
                        (perm_combs[:, 0] <= perm_combs[:, 1]) &
                        (perm_combs[:, 0] <= perm_combs[:, 2])]         
    
    bilength = perm_combs.shape[0]
    res_stack1 = np.zeros((bilength,imshape[0],imshape[1]),dtype = 'int8')
    for indval in tqdm(range(bilength)):
        i, j, k, l = perm_combs[indval]
        res_stack1[indval,:,:] = (imgs1a[i-1,:,:] + imgs1a[j-1,:,:]) > (imgs1a[k-1,:,:] + imgs1a[l-1,:,:])

    return res_stack1
def biconv4(imgs, n = 8, n2=20):
    comN = 4
    #Compare to average and also compare combinations of images
    imshape = imgs[0].shape
    imgs1a = np.zeros((n2,imshape[0],imshape[1]))
    imgs1b = np.zeros((n2,imshape[0],imshape[1]))
    for a in range(n2):
        imgs1a[a,:,:]  = imgs[a,:,:]
        
    avg_img = imgs1a.mean(axis=(0))
    for b in range(n):
        imgs1b[b,:,:] = imgs1a[b,:,:] > avg_img
        
    imgs1c = np.zeros((n,imshape[0],imshape[1]))    
    for a in range(n):
        imgs1c[a,:,:]  = imgs[a,:,:]
    combs = list(itt.combinations(range(1, n + 1), comN))
    perm_combs = []

    for comb in combs:
        perm_combs.extend(itt.permutations(comb))

    perm_combs = np.array(sorted(perm_combs))
   # Remove unwanted permutations
    perm_combs = perm_combs[(perm_combs[:, 2] <= perm_combs[:, 3]) &
                        (perm_combs[:, 0] <= perm_combs[:, 1]) &
                        (perm_combs[:, 0] <= perm_combs[:, 2])]         
    
    bilength = perm_combs.shape[0]
    res_stack1 = np.zeros((bilength,imshape[0],imshape[1]),dtype = 'int8')
    for indval in tqdm(range(bilength)):
        i, j, k, l = perm_combs[indval]
        res_stack1[indval,:,:] = (imgs1c[i-1,:,:] + imgs1c[j-1,:,:]) > (imgs1c[k-1,:,:] + imgs1c[l-1,:,:])

    res = np.concatenate((res_stack1,imgs1b))

    return res

@numba.jit(nopython=True)
def bi_pix(Gi,y,n, xLim, maskR, xOffset1, xOffset2):
    max_cor = 0.0
    max_index = -1
    max_mod = [0,0]
    
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = maskR[:,y,xi]
        chkres = Gi-Gt
        chk = np.sum(chkres == 0)/n
        if(chk > max_cor):
            max_index = xi
            max_cor = chk
         
    Gup = maskR[:,y-1, max_index]
    chkres = Gi-Gup
    chk = np.sum(chkres == 0)/n
    if(chk > max_cor):
        max_mod  = [-1,0]
        max_cor = chk
    
    Gdn = maskR[:,y+1, max_index]
    chkres = Gi-Gdn
    chk = np.sum(chkres == 0)/n
    if(chk > max_cor):
        max_mod  = [1,0]
        max_cor = chk
        
    return max_index,max_cor,max_mod
@numba.jit(nopython=True)
def ncc_pix(Gi,y,n, xLim, maskR, xOffset1, xOffset2):
    max_cor = 0.0
    max_index = -1
    max_mod = [0,0]
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


    
@numba.jit(nopython=True)
def bi_pix2(Gi,y,n, xLim, maskR, xOffset1, xOffset2, Gi2, images2):
    max_cor = 0.0
    max_index = -1
    max_mod = [0,0]
    poi_list = []
    #Initial pass
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = maskR[:,y,xi]
        chkres = Gi-Gt
        chk = np.sum(chkres == 0)/n
        if(chk > max_cor):
            max_index = xi
            max_cor = chk
    
    #second pass to find additional points of interest        
    for a in range(max_index,xLim-xOffset2):
        Gt = maskR[:,y,a]
        chkres = Gi-Gt
        chk = np.sum(chkres == 0)/n
        if(chk >= max_cor):
            poi_list.append(a)
    
    
    #third pass to verify points of interest with ncc  
    n2 = len(images2)      
    agi = np.sum(Gi2)/n2
    val_i = np.sum((Gi2-agi)**2)        
    max_cor_ncc = 0.0
    for b in poi_list:

        Gt = images2[:,y,b]
        agt = np.sum(Gt)/n2       
        val_t = np.sum((Gt-agt)**2) 
        if(val_i > float_epsilon and val_t > float_epsilon): 
            cor = np.sum((Gi2-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))              
            if cor > max_cor_ncc:
                max_cor_ncc = cor
                max_index = b      
    return max_index,max_cor_ncc,max_mod


def comcor1(res_list, entry_val, threshold):
    remove_flag = False
    pos_remove = 0
    entry_flag = False
    counter = 0

    if(entry_val[1] < 0 or entry_val[2] < threshold):

        return pos_remove,remove_flag,entry_flag
    for i in range(len(res_list)):       
        
        if(res_list[i][1] == entry_val[1] and res_list[i][3][0] - entry_val[3][0] < float_epsilon and
           res_list[i][3][1] - entry_val[3][1] < float_epsilon):
            #duplicate found, check correlation values and mark index for removal
            remove_flag = (res_list[i][2] < entry_val[2])
            
            pos_remove = i
            break
        else:
            counter+=1
    #end of list reached, no duplicates found, entry is valid
    if(counter == len(res_list)):
        entry_flag = True
    return pos_remove,remove_flag,entry_flag
def cormap(rect_res, img_ref):
    imshape = img_ref.shape
    resmap = np.zeros(imshape)
    for a in range(len(rect_res)):
        b = rect_res[a]
        
        for q in b:
            xL = q[0]
            y = q[4]
            xR = q[1]
            subx = q[3][1]
            suby = q[3][0]
            cor = q[2]
            resmap[y,xL] = cor
    
    return resmap
            
def run_bicos():
    #load images
    imgFolder = './test_data/testset1/bulb/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgs1,imgs2 = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    col_refL, col_refR = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd,colorIm = True)
    #apply filter
    thresh1 = 30
    imgs1 = np.asarray(scr.mask_inten_list(imgs1,thresh1))
    imgs2 = np.asarray(scr.mask_inten_list(imgs2,thresh1))
    #load matrices
    mat_folder = './test_data/testset1/matrices/'
    kL, kR, r, t = scr.load_mats(mat_folder) 
    f = np.loadtxt(mat_folder + 'f.txt', delimiter = ' ', skiprows = 2)
    #rectify images
    v,w, H1, H2 = scr.rectify_pair(imgs1[0], imgs2[0], f)
    imgs1,imgs2 = scr.rectify_lists(imgs1,imgs2,f)
    imgs1 = np.asarray(imgs1)
    imgs2 = np.asarray(imgs2)
    imshape = imgs1[0].shape
    n = 8
    #binary conversion
    imgs1a = biconv2(imgs1, n = n)
    imgs2a = biconv2(imgs2, n = n)
    #take first n images
    imgs1b = np.zeros((n,imshape[0],imshape[1]))
    imgs2b = np.zeros((n,imshape[0],imshape[1]))
    for b in range(n):
        imgs1b[b,:,:] = imgs1[b,:,:]
        imgs2b[b,:,:] = imgs2[b,:,:]
    #Take left and compare to right side to find matches
    offset = 10
    rect_res = []
    xLim = imshape[1]
    yLim = imshape[0]

    n2 = len(imgs1a)
    print(n2)
    cor_thresh = 0.9
    for y in tqdm(range(offset, yLim-offset)):
        res_y = []
        for x in range(offset, xLim-offset):
            Gi = imgs1a[:,y,x].astype('uint8')
            Gi2 = imgs1b[:,y,x]
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                x_match,cor_val,subpix= bi_pix2(Gi,y,n2, xLim, imgs2a, offset, offset, Gi2, imgs2b)
                pos_remove, remove_flag, entry_flag = comcor1(res_y,
                                                                  [x,x_match, cor_val, subpix, y], cor_thresh)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, subpix, y])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, subpix, y])
        
        rect_res.append(res_y)  
    
 

    cor_list = []
    hL_inv = np.linalg.inv(H1)
    hR_inv = np.linalg.inv(H2)
    ptsL = []
    ptsR = []
    for a in range(len(rect_res)):
        b = rect_res[a]
        
        for q in b:
            xL = q[0]
            y = q[4]
            xR = q[1]
            subx = q[3][1]
            suby = q[3][0]
            xL_u = (hL_inv[0,0]*xL + hL_inv[0,1] * (y+suby) + hL_inv[0,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * (y+suby)  + hL_inv[2,2])
            yL_u = (hL_inv[1,0]*xL + hL_inv[1,1] * (y+suby)  + hL_inv[1,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * (y+suby)  + hL_inv[2,2])
            xR_u = (hR_inv[0,0]*(xR+subx) + hR_inv[0,1] * (y+suby)  + hR_inv[0,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * (y+suby)  + hR_inv[2,2])
            yR_u = (hR_inv[1,0]*(xR+subx) + hR_inv[1,1] * (y+suby)  + hR_inv[1,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * (y+suby)  + hR_inv[2,2])
            ptsL.append([xL_u,yL_u])
            ptsR.append([xR_u,yR_u])
            cor_list.append(q[2])
          
    col_ptsL = np.around(ptsL,0).astype('uint16')
    col_ptsR = np.around(ptsR,0).astype('uint16')  
    print(np.min(cor_list))
    print(np.max(cor_list))
    #col_arr = scr.get_color(col_refL, col_refR, col_ptsL, col_ptsR)      
    col_arr = scr.create_colcor_arr(cor_list, cor_thresh)
    tri_res = scr.triangulate_list(ptsL,ptsR, r, t, kL, kR)
    
    scr.convert_np_ply(np.asarray(tri_res), col_arr,"tbicos2.ply")
    cor_map = cormap(rect_res, imgs1[0])
    plt.imshow(cor_map)
    plt.show()
    filmap = sig.medfilt2d(cor_map, 5)
    plt.imshow(filmap)
    plt.show()
run_bicos()
def t1():
    a = [[1,1,0.91,[0,0],0],[0,0,0.92,[0,0],0]]
    i,j,k = comcor1(a,[2,2,0.91,[0,0],0],0.9)
    print(i)
    print(j)
    print(k)
#t1()
