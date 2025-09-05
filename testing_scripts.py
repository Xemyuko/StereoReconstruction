# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:09:26 2023

@author: Admin
"""

import numpy as np
import scripts as scr
import matplotlib.pyplot as plt
import ncc_core as ncc
from tqdm import tqdm
import cv2
import json
import numba

import threading as thr
from numba import cuda as cu
import time
import os
import random
import confighandler as chand
import scipy.signal as sig
from scipy.interpolate import Rbf
import scipy.interpolate
import scipy.linalg as sclin
import tkinter as tk
import inspect
import csv
import point_cloud_utils as pcu
import itertools as itt
import sewar.full_ref as swr
#used for comparing floating point numbers to avoid numerical errors
float_epsilon = 1e-9

def recon_comp_data_gen():

    #load test images
    folder1 = './test_data/denoise_unet/trec_outputs1/'
    folder2 = './test_data/denoise_unet/trec_reference1/'
    imgsL1,imgsR1= scr.load_images_1_dir(folder1,'cam1', 'cam2', ext = '.jpg')
    imgsL2,imgsR2= scr.load_images_1_dir(folder2,'cam1', 'cam2', ext = '.jpg')
    #load matrices
    matFolder = './test_data/denoise_unet/matrices/'
    f_file = 'fund.txt'
    kL, kR, r, t = scr.load_mats(matFolder)
    F = np.loadtxt(matFolder + f_file, delimiter = ' ', skiprows = 2)
    #ncc correlate points in test images
    offset = 10
    ptsL1,ptsR1 = ncc.cor_pts_pix(imgsL1, imgsR1, kL, kR, F, offset)
    #ncc correlate points in reference images
    ptsL2,ptsR2 = ncc.cor_pts_pix(imgsL2, imgsR2, kL, kR, F, offset)
    
    #triangulate points
    tri1 = scr.triangulate_list(ptsL1, ptsR1, r, t, kL, kR)
    tri2 = scr.triangulate_list(ptsL2, ptsR2, r, t, kL, kR)
    #create data lists with point information
    d1 = []
    for i in range(len(tri1)):
        ent = [ptsL1[i][0],ptsL1[i][1], ptsR1[i][0], ptsR1[i][1], tri1[i][0], tri1[i][1], tri1[i][2]]
        d1.append(ent)
    d2 = []
    for i in range(len(tri2)):
        ent = [ptsL2[i][0],ptsL2[i][1], ptsR2[i][0], ptsR2[i][1], tri2[i][0], tri2[i][1], tri2[i][2]]
        d2.append(ent)
    d1 = np.asarray(d1)
    print(d1.shape)
    print(d1.dtype)
    d2 = np.asarray(d2)
    np.savetxt('d1.txt',d1, delimiter = ' ')
    np.savetxt('d2.txt',d2, delimiter = ' ')


#recon_comp_data_gen()
    
@numba.jit(nopython=True)
def boost_comp(i, ptsL1x,ptsL1y,ptsL2x,ptsL2y, ptsR1x,ptsR1y,ptsR2x,ptsR2y, checkval):
    print(range(i - checkval, i + checkval))
    for j in range(i - checkval, i + checkval):
        if(ptsL1x[i] - ptsL2x[j] < float_epsilon and ptsL1y[i] - ptsL2y[j] < float_epsilon):
            a = np.sqrt((ptsR1x[i]-ptsR2x[j])**2 + (ptsR1y[i]-ptsR2y[j])**2)
            return a
    return -9000.0
    
   
def data_comp():
    d1 = np.loadtxt('d1.txt', delimiter = ' ')
    d2 = np.loadtxt('d2.txt', delimiter = ' ')
    ptsL1x = []
    ptsL1y = []
    ptsR1x = []
    ptsR1y = []
    tri1x = []
    tri1y = []
    tri1z = []
    print(d1[0])
    for i in d1:
        ptsL1x.append(i[0])
        ptsL1y.append(i[1])
        ptsR1x.append(i[2])
        ptsR1y.append(i[3])
        tri1x.append(i[4])
        tri1y.append(i[5])
        tri1z.append(i[6])
    ptsL2x = []
    ptsL2y = []
    ptsR2x = []
    ptsR2y = []
    tri2x = []
    tri2y = []
    tri2z = []
    for i in d2:
        ptsL2x.append(i[0])
        ptsL2y.append(i[1])
        ptsR2x.append(i[2])
        ptsR2y.append(i[3])
        tri2x.append(i[4])
        tri2y.append(i[5])
        tri2z.append(i[6])
    diff_pts = []
    checkval = 1000
    for i in tqdm(range(checkval,len(ptsL1x) - checkval)): 
        for j in range(i - checkval, i + checkval):
            if(ptsL1x[i] - ptsL2x[j] < float_epsilon and ptsL1y[i] - ptsL2y[j] < float_epsilon):
                a = np.sqrt((ptsR1x[i]-ptsR2x[j])**2 + (ptsR1y[i]-ptsR2y[j])**2)
                diff_pts.append(a)
    diff_pts = np.asarray(diff_pts)
    print(np.average(diff_pts))
    print(len(diff_pts))
    

data_comp()

def test_img_comp():
    folder0 = './test_data/denoise_unet/trec_inputs1/'
    folder1 = './test_data/denoise_unet/trec_outputs1/'
    folder2 = './test_data/denoise_unet/trec_reference1/'
    imgs0= scr.load_all_imgs_1_dir(folder0)
    imgs1= scr.load_all_imgs_1_dir(folder1)
    imgs2 = scr.load_all_imgs_1_dir(folder2)
    im_n = 0
    im1 = imgs1[im_n]
    im2 = imgs2[im_n]
    im3 = imgs2[im_n+1]
    im4 = scr.boost_zone(imgs0[im_n], 50, 1, 1, 1, 1)
    s1,diff1 = scr.ssim_compare(im1, im2)
    sr,diffr = scr.ssim_compare(im3, im2)
    sb,diffb = scr.ssim_compare(im4, im2)
    
    scr.dptle(diff1, 'Unet vs ref', cmap = 'gray')
    scr.dptle(diffr, 'ref vs diff pattern', cmap = 'gray')
    scr.dptle(diffb, 'contrast boost vs ref', cmap = 'gray')
    s2 = swr.rmse(im1,im2)
    s2r = swr.rmse(im3,im2)
    s2b = swr.rmse(im4,im2)
    s3 = swr.psnr(im1,im2)
    s3r = swr.psnr(im3,im2)
    s3b = swr.psnr(im4,im2)
    s4 = np.real(swr.msssim(im1,im2))
    s4r = np.real(swr.msssim(im3,im2))
    s4b = np.real(swr.msssim(im4,im2))
    s5 = swr.uqi(im1,im2)
    s5r = swr.uqi(im3,im2)
    s5b = swr.uqi(im4,im2)
    print('unet-ref SSIM = ' + str(s1))
    print('pat-ref SSIM = ' + str(sr))
    print('cont-ref SSIM = ' + str(sb))
    print('##########################')
    print('unet-ref RMSE = ' + str(s2))
    print('pat-ref RMSE = ' + str(s2r))
    print('cont-ref RMSE = ' + str(s2b))
    print('##########################')
    print('unet-ref PSNR = ' + str(s3))
    print('pat-ref PSNR = ' + str(s3r))
    print('cont-ref PSNR = ' + str(s3b))
    print('##########################')
    print('unet-ref MS-SSIM = ' + str(s4))
    print('pat-ref MS-SSIM = ' + str(s4r))
    print('cont-ref MS-SSIM = ' + str(s4b))
    print('##########################')
    print('unet-ref UQI = ' + str(s5))
    print('pat-ref UQI = ' + str(s5r))
    print('cont-ref UQI = ' + str(s5b))
    print('##########################')

def test_pcu():
    f1 = 'recon_ref1.ply'
    f2 = 'recon_set1.ply'
    f3 = 'recon_set1_cb.ply'
    f4 = 'bulb12ncc.ply'
    f5 = 'bulb20ncc.ply'
    p1 = pcu.load_mesh_v(f1)
    p2 = pcu.load_mesh_v(f2)
    p3 = pcu.load_mesh_v(f3)
    p4 = pcu.load_mesh_v(f4)
    p5 = pcu.load_mesh_v(f5)
    cdr = pcu.chamfer_distance(p1, p1)
    cd1 = pcu.chamfer_distance(p1, p2)
    cd2 = pcu.chamfer_distance(p1, p3)
    cd3 = pcu.chamfer_distance(p2, p3)
    cd4 = pcu.chamfer_distance(p1, p4)
    cd5 = pcu.chamfer_distance(p4, p5)
    print('ref-ref cd: ' + str(cdr))
    print('ref-unet cd: ' + str(cd1))
    print('ref-contrast cd: ' + str(cd2))
    print('unet-contrast cd: ' + str(cd3))
    print('ref-diff cd: ' + str(cd4))
    print('diff-diff1 cd: ' + str(cd5))
    print('####################')
    hdr = pcu.hausdorff_distance(p1, p1)
    hd1 = pcu.hausdorff_distance(p1, p2)
    hd2 = pcu.hausdorff_distance(p1, p3)
    hd3 = pcu.hausdorff_distance(p2, p3)    
    hd4 = pcu.hausdorff_distance(p1, p4)
    hd5 = pcu.hausdorff_distance(p4, p5)
    print('ref-ref hd: ' + str(hdr))
    print('ref-unet hd: ' + str(hd1))
    print('ref-contrast hd: ' + str(hd2))
    print('unet-contrast hd: ' + str(hd3))
    print('ref-diff hd: ' + str(hd4))
    print('diff-diff1 cd: ' + str(hd5))

def test_unrect():
    #load images
    #load image pair
    folder1 = './test_data/250221_Cudatest/pos7/'
    #folder1 = './test_data/testset1/bulb-multi/b1/'
    
    imgL,imgR = scr.load_first_pair_1_dir(folder1, 'cam1', 'cam2', ext = '.jpg')
    
    #load matrices
    #load matrices
    matFolder = './test_data/testset1/matrices/'
    f_file = 'f.txt'
    kL, kR, r, t = scr.load_mats(matFolder)
    F = np.loadtxt(matFolder + f_file, delimiter = ' ', skiprows = 2)
    #rectify images
    rectL,rectR,H1, H2 = scr.rectify_pair(imgL,imgR, F)
    #inverse rectification homography
    hL_inv = np.linalg.inv(H1)
    hR_inv = np.linalg.inv(H2)
    
    
    #unrectify images
    imshape = imgL.shape
    revshape = (imshape[1],imshape[0])
    img1 = cv2.warpPerspective(rectL, hL_inv, revshape)
    img2 = cv2.warpPerspective(imgR, hR_inv, revshape)
    scr.display_stereo(imgL,imgR)
    scr.display_stereo(rectL,rectR)
    scr.display_stereo(img1,img2)
    scr.display_4_comp(imgL,imgR, img1, img2)





def spat_extract(img):
    #pulls 8 immediate neighbours + 16 next neighbours + 32 next neighbors for 49 intensity points per pixel
    #input: img
    #output: 49 images in stack
    offset = 3
    #create output image array stack
    #loop through image pixel by pixel, with offsets at each side. 
    #get values at each neighboring point of interest
    #write values to locations in output
    imshape = img.shape
    n = 49
    res = np.zeros((n,imshape[0],imshape[1]), dtype = img.dtype)
    for i in range(offset,imshape[0] - offset):
        for j in range(offset,imshape[1] - offset):
            #assign central pixel
            res[0,i,j] = img[i,j]
            #assign cardinal directions, first layer, NSEW
            res[1,i,j] = img[i-1,j]
            res[2,i,j] = img[i+1,j]
            res[3,i,j] = img[i,j-1]
            res[4,i,j] = img[i,j+1]
            #first layer diagonals
            res[5,i,j] = img[i-1,j-1]
            res[6,i,j] = img[i+1,j+1]
            res[7,i,j] = img[i+1,j-1]
            res[8,i,j] = img[i-1,j+1]
            #second layer cardinals
            res[9,i,j] = img[i-2,j]
            res[10,i,j] = img[i+2,j]
            res[11,i,j] = img[i,j-2]
            res[12,i,j] = img[i,j+2]
            #second layer diagonals
            res[13,i,j] = img[i-2,j-2]
            res[14,i,j] = img[i+2,j-2]
            res[15,i,j] = img[i-2,j+2]
            res[16,i,j] = img[i+2,j+2]
            #second layer fills
            res[17,i,j] = img[i-2,j-1]
            res[18,i,j] = img[i+2,j-1]
            res[19,i,j] = img[i-1,j-2]
            res[20,i,j] = img[i-1,j+2]
            res[21,i,j] = img[i-2,j+1]
            res[22,i,j] = img[i+2,j+1]
            res[23,i,j] = img[i+1,j-2]
            res[24,i,j] = img[i+1,j+2]
            
            #third layer cardinals
            res[25,i,j] = img[i-3,j]
            res[26,i,j] = img[i+3,j]
            res[27,i,j] = img[i,j-3]
            res[28,i,j] = img[i,j+3]
            #third layer diagonals
            res[29,i,j] = img[i-3,j-3]
            res[30,i,j] = img[i+3,j-3]
            res[31,i,j] = img[i-3,j+3]
            res[32,i,j] = img[i+3,j+3]
            #third layer fills
            res[33,i,j] = img[i-3,j-2]
            res[34,i,j] = img[i+3,j-2]
            res[35,i,j] = img[i-2,j-3]
            res[36,i,j] = img[i-2,j+3]
            res[37,i,j] = img[i-3,j+2]
            res[38,i,j] = img[i+3,j+2]
            res[39,i,j] = img[i+2,j-3]
            res[40,i,j] = img[i+2,j+3]
            
            res[41,i,j] = img[i-3,j-1]
            res[42,i,j] = img[i+3,j-1]
            res[43,i,j] = img[i-1,j-3]
            res[44,i,j] = img[i-1,j+3]
            res[45,i,j] = img[i-3,j+1]
            res[46,i,j] = img[i+3,j+1]
            res[47,i,j] = img[i+1,j-3]
            res[48,i,j] = img[i+1,j+3]
            
            
    return res

def point_marker(img, x,y):
    img[y,x] = 0
    img[y-1,x] = 240
    img[y+1,x] = 240
    img[y,x-1] = 240
    img[y,x+1] = 240
    img[y-1,x-1] = 120
    img[y+1,x+1] = 120
    img[y+1,x-1] = 120
    img[y-1,x+1] = 120
    return img



@numba.jit(nopython=True)
def ncc_pix(Gi,y,n, xLim, maskR, xOffset1, xOffset2):
    max_cor = 0.0
    max_mod = [0,0]
    max_index = -1
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
    return max_index,max_cor, max_mod



def comcor1(res_list, entry_val, threshold):
    remove_flag = False
    pos_remove = 0
    entry_flag = False
    counter = 0
    if(entry_val[1] < 0 or entry_val[2] < threshold):
        return pos_remove,remove_flag,entry_flag
    for i in range(len(res_list)):       
        
        if(res_list[i][1] == entry_val[1]):
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
def ncc_pix_precalc(Gi,x,y,n, xLim, maskR, xOffset1, xOffset2, resL,resR):
    max_cor = 0.0
    max_mod = [0,0]
    max_index = -1
    agi = resL[y,x,0]
    val_i = resL[y,x,1]
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = maskR[:,y,xi]
        agt = resR[y,xi,0]       
        val_t = resR[y,xi,1]
        if(val_i > float_epsilon and val_t > float_epsilon): 
            cor = np.sum((Gi-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))              
            if cor > max_cor:
                max_cor = cor
                max_index = xi
    #search surroundings of found best match
    Gup = maskR[:,y-1, max_index]
    agup = resR[y-1,max_index,0]
    val_up = resR[y-1,max_index,1]
    if(val_i > float_epsilon and val_up > float_epsilon): 
        cor = np.sum((Gi-agi)*(Gup - agup))/(np.sqrt(val_i*val_up))              
        if cor > max_cor:
           max_cor = cor
           max_mod = [-1,0]
    
    Gdn = maskR[:,y+1, max_index]
    agdn = resR[y+1,max_index,0]
    val_dn = resR[y+1,max_index,1]
    if(val_i > float_epsilon and val_dn > float_epsilon): 
        cor = np.sum((Gi-agi)*(Gdn - agdn))/(np.sqrt(val_i*val_dn))              
        if cor > max_cor:
            max_cor = cor
            max_mod = [1,0]  
    return max_index,max_cor, max_mod
def calc_ncc_components(imgsL, imgsR, offset = 10):
    image_size = imgsL[0].shape
    n = len(imgsL)
    resL = np.zeros((image_size[0], image_size[1], 2))
    resR = np.zeros((image_size[0], image_size[1], 2))
    xLim = image_size[1]
    yLim = image_size[0]
    for i in tqdm(range(0, yLim)):
        for j in range(0, xLim):
            
            gL = imgsL[:,i,j]
            gR = imgsR[:,i,j]
            
            agL = np.sum(gL)/n    
            if agL > 0:
                val_L = np.sum((gL-agL)**2)
            else:
                val_L = 0
            agR = np.sum(gR)/n        
            val_R = np.sum((gR-agR)**2)
            resL[i,j,0] = agL
            resL[i,j,1] = val_L
            resR[i,j,0] = agR
            resR[i,j,1] = val_R
            
    return resL,resR
            #if(val_i > float_epsilon and val_t > float_epsilon): 
               # cor = np.sum((Gi-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))


               
def convert_stack1(stack):
    suv = 0
    for i in stack:
        suv += i
    suv/=len(stack)
    
    for a in range(1, len(stack)):
        valP = stack[a-1]
        valA = stack[a]
        diffA = valA-valP
        
    return 0


        
def test_sift_spat():
    #load image pair
    folder1 = './test_data/250221_Cudatest/pos7/'
    #folder1 = './test_data/testset1/bulb-multi/b1/'
    imgLc,imgRc = scr.load_images_1_dir(folder1, 'cam1', 'cam2', ext = '.jpg', colorIm = True)
    imgL,imgR = scr.load_images_1_dir(folder1, 'cam1', 'cam2', ext = '.jpg', colorIm = False)
    #load camera intrinsics
    mat_folder = './test_data/testset1/matrices/'
    kL, kR, R, t = scr.load_mats(mat_folder)
    #apply sift to find feature matches
    #calculate F from feature matches
    pts1,pts2,col,F = scr.feature_corr(imgLc[1],imgRc[1])
    #draw found matching points on stereo
    scr.mark_points(imgLc[0],imgRc[0],pts1,pts2,size = 20,showBox = False)
    v,w, H1, H2 = scr.rectify_pair(imgL[0], imgR[0], F)
    imgLc,imgRc = scr.rectify_lists(imgLc,imgRc,F)
    imgsL,imgR = scr.rectify_lists(imgL,imgR,F)
    imgLT = imgL[1]
    imgRT = imgR[1]
    '''
    for a  in pts1:
        imgLT = point_marker(imgLT, a[0], a[1])
    for a  in pts2:
        imgRT = point_marker(imgRT, a[0], a[1])
    '''
    plt.imshow(imgLT)
    plt.show()
    tL= spat_extract(imgLT)
    tR = spat_extract(imgRT)
    tLa = spat_extract(imgL[0])
    tRa = spat_extract(imgR[0])
    preL,preR = calc_ncc_components(tL,tR)
    preLa,preRa = calc_ncc_components(tLa,tRa)
    offset = 10
    imshape = tL[0].shape
    xLim = imshape[1]
    yLim = imshape[0]
    n2 = len(tL)
    cor_thresh = 0.9
    rect_res = []  
    for y in tqdm(range(offset, yLim-offset)):
        res_y = []
        for x in range(offset, xLim-offset):
            Gi = tL[:,y,x]
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                x_match,cor_val, max_mod= ncc_pix_precalc(Gi,x,y,n2, xLim, tR, offset, offset, preL,preR)
                pos_remove, remove_flag, entry_flag = comcor1(res_y,
                                                                  [x,x_match, cor_val, y, max_mod], cor_thresh)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, y, max_mod])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val,y, max_mod])
        
        rect_res.append(res_y)
    #unrectify points and triangulate
    hL_inv = np.linalg.inv(H1)
    hR_inv = np.linalg.inv(H2)
    ptsL = []
    ptsR = []
    for a in range(len(rect_res)):
        b = rect_res[a]
        for q in b:
            xL = q[0]
            y = q[3]
            xR = q[1]
            subx = q[4][1]
            suby = q[4][0]
            
            xL_u = (hL_inv[0,0]*xL + hL_inv[0,1] * (y+suby) + hL_inv[0,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * (y+suby)  + hL_inv[2,2])
            yL_u = (hL_inv[1,0]*xL + hL_inv[1,1] * (y+suby)  + hL_inv[1,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * (y+suby)  + hL_inv[2,2])
            xR_u = (hR_inv[0,0]*(xR+subx) + hR_inv[0,1] * (y+suby)  + hR_inv[0,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * (y+suby)  + hR_inv[2,2])
            yR_u = (hR_inv[1,0]*(xR+subx) + hR_inv[1,1] * (y+suby)  + hR_inv[1,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * (y+suby)  + hR_inv[2,2])
            ptsL.append([xL_u,yL_u])
            ptsR.append([xR_u,yR_u])
            
             
    col_ptsL = np.around(ptsL,0).astype('uint16')
    col_ptsR = np.around(ptsR,0).astype('uint16')
    
    col_arr = scr.gen_color_arr_black(len(ptsL))
    tri_res = scr.triangulate_list(ptsL,ptsR,R, t, kL, kR)
    scr.convert_np_ply(np.asarray(tri_res), col_arr,'test_sift_spat.ply')


def precalc_ncc_test():
    #load images 
    #load matrices
    start = time.time()
    imgFolder = './test_data/testset1/bulb-multi/b1/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgs1,imgs2 = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    image_size = imgs1[0].shape 
    mat_folder = './test_data/testset1/matrices/'
    kL, kR, R, t = scr.load_mats(mat_folder) 
    col_refL, col_refR = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd,colorIm = True)
    f = np.loadtxt(mat_folder + 'f.txt', delimiter = ' ', skiprows = 2)
    #rectify images
    v,w, H1, H2 = scr.rectify_pair(imgs1[0], imgs2[0], f)
    ims1,ims2 = scr.rectify_lists(col_refL,col_refR,f)
    imgs1,imgs2 = scr.rectify_lists(imgs1,imgs2,f)
    im1 = ims1[0]
    im2 = ims2[0]
    imshape = imgs1[0].shape
    imgs1 = np.asarray(imgs1)
    imgs2 = np.asarray(imgs2)
    end = time.time()
    print('Load Time:')
    print(end - start) 
    #run search for points
    #store total points found and list of points
    n2 = len(imgs1)
    cor_thresh = 0.0
    offset = 10
    rect_res = []
    xLim = imshape[1]
    yLim = imshape[0]
    
    
    
    for y in tqdm(range(offset, yLim-offset)):
        res_y = []
        for x in range(offset, xLim-offset):
            Gi = imgs1[:,y,x]
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                x_match,cor_val, max_mod= ncc_pix(Gi,y,n2, xLim, imgs2, offset, offset)
                pos_remove, remove_flag, entry_flag = comcor1(res_y,
                                                                  [x,x_match, cor_val, y, max_mod], cor_thresh)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, y, max_mod])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val,y, max_mod])
        
        rect_res.append(res_y)
    rect_res2 = []    
    resL,resR = calc_ncc_components(imgs1,imgs2)
    for y in tqdm(range(offset, yLim-offset)):
        res_y = []
        for x in range(offset, xLim-offset):
            Gi = imgs1[:,y,x]
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                x_match,cor_val, max_mod= ncc_pix_precalc(Gi,x,y,n2, xLim, imgs2, offset, offset, resL,resR)
                pos_remove, remove_flag, entry_flag = comcor1(res_y,
                                                                  [x,x_match, cor_val, y, max_mod], cor_thresh)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, y, max_mod])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val,y, max_mod])
        
        rect_res2.append(res_y)
  

def comp_tri():
    #load matrices
    kL, kR, r, t = scr.load_mats("./test_data/testset1/matrices/")
    
    pt1 = [10.0,20.0]
    pt2 = [150.0,250.0]
    Al = np.c_[kL, np.asarray([[0],[0],[0]])]


    RT = np.c_[r, t]
    Ar = kR @ RT
    
    sol0 = pt1[1] * Al[2,:] - Al[1,:]
    sol1 = -pt1[0] * Al[2,:] + Al[0,:]
    sol2 = pt2[1] * Ar[2,:] - Ar[1,:]
    sol3 = -pt2[0] * Ar[2,:] + Ar[0,:]
    
    
    solMat = np.stack((sol0,sol1,sol2,sol3))

    #Apply SVD to solution matrix to find triangulation
    U,s,vh = np.linalg.svd(solMat,full_matrices = True)

    Q = vh[3,:]

    Q /= Q[3]
    print(str(Q[0]) + ' ' + str(Q[1]) + ' ' + str(Q[2]))



def run_sift():
    #load images in color
    imgFolder = './test_data/testset0/moon2/'
    imgs = scr.load_all_imgs_1_dir(imgFolder)
    img1 = imgs[0]
    img2 = imgs[1]
    scr.display_stereo(img1,img2)
    img1 = scr.boost_zone(img1,1.5,1,1,1,1)
    img2 = scr.boost_zone(img2,1.5,1,1,1,1)
    scr.display_stereo(img1,img2)
    #run sift
    thresh = 0.4
    #identify feature points to correlate
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    sp1, des1 = sift.detectAndCompute(img1,None)
    sp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    pts1 = []
    pts2 = []

    for i,(m,n) in enumerate(matches):
        if m.distance < thresh*n.distance:
            pts2.append(sp2[m.trainIdx].pt)
            pts1.append(sp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    #mark points on stereo image
    scr.mark_points(imgs[0],imgs[1], pts1,pts2, size = 20)





def biconv3(imgs, n = 8, comN = 4):
    imshape = imgs[0].shape
    imgs1a = np.zeros((n,imshape[0],imshape[1]))
    for a in range(n):
        imgs1a[a,:,:]  = imgs[a,:,:]
    avg_img = imgs1a.mean(axis=(0))
    imgs1b = np.zeros((n,imshape[0],imshape[1]))
    for i in range(1,n):
        imgs1b[i-1,:,:] = imgs[i-1,:,:] + imgs[i,:,:]
        
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
    

def biconv2(imgs, n = 8):
    imshape = imgs[0].shape
    imgs1a = np.zeros((n,imshape[0],imshape[1]))
    imgs1b = np.zeros((n,imshape[0],imshape[1]))
    for a in range(n):
        imgs1a[a,:,:]  = imgs[a,:,:]
        
    avg_img = imgs1a.mean(axis=(0))
    for b in range(n):
        imgs1b[b,:,:] = imgs1a[b,:,:] > avg_img
    return imgs1b
def biconv1(imgs, n = 8, comN = 4):
    
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

@numba.jit(nopython=True)
def bi_pix(Gi,y,n, xLim, maskR, xOffset1, xOffset2):
    max_cor = 0.0
    max_index = -1
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = maskR[:,y,xi]
        chkres = Gi-Gt
        chk = np.sum(chkres == 0)/n
        if(chk > max_cor):
            max_index = xi
            max_cor = chk
    return max_index,max_cor




def lim1():
    #test limiting the epipolar lines 

    #load images 
    #load matrices
    imgFolder = './test_data/testset1/bulb-multi/b1/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgs1,imgs2 = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    image_size = imgs1[0].shape 
    mat_folder = './test_data/testset1/matrices/'
    kL, kR, R, t = scr.load_mats(mat_folder) 
    col_refL, col_refR = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd,colorIm = True)
    f = np.loadtxt(mat_folder + 'f.txt', delimiter = ' ', skiprows = 2)
    #rectify images
    v,w, H1, H2 = scr.rectify_pair(imgs1[0], imgs2[0], f)
    ims1,ims2 = scr.rectify_lists(col_refL,col_refR,f)
    imgs1,imgs2 = scr.rectify_lists(imgs1,imgs2,f)
    im1 = ims1[0]
    im2 = ims2[0]
    imshape = imgs1[0].shape
    imgs1 = np.asarray(imgs1)
    imgs2 = np.asarray(imgs2)
    #run search for points
    #store total points found and list of points
    n2 = len(imgs1)
    cor_thresh = 0.0
    offset = 10
    rect_res = []
    xLim = imshape[1]
    yLim = imshape[0]
    print(xLim)
    
    
    
    for y in tqdm(range(offset, yLim-offset)):
        res_y = []
        for x in range(offset, xLim-offset):
            Gi = imgs1[:,y,x]
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                x_match,cor_val= ncc_pix(Gi,y,n2, xLim, imgs2, offset, offset)
                pos_remove, remove_flag, entry_flag = comcor1(res_y,
                                                                  [x,x_match, cor_val, y], cor_thresh)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, y])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val,y])
        
        rect_res.append(res_y)

    
    #set search range and storage lists
    xR_list = []
    xL_list = []
    y_list = []
    srch_rng = 50
    
    #loop through matched results
    for a in range(len(rect_res)):
        b = rect_res[a]
        for q in b:
            xL = q[0]
            y = q[3]
            xR = q[1]
            xL_list.append(xL)
            xR_list.append(xR)
            y_list.append(y)
            
    print(len(xR_list))
    print(xL_list[0])
    print(xR_list[0])
    print(max(xR_list[0]-srch_rng,offset))
    print(max(xR+srch_rng,offset))
    print(xLim - xR+srch_rng)
    res2 = [] 
    start = time.time()
    for a in tqdm(range(len(xR_list))):
        xL = xL_list[a]
        y = y_list[a]
        xR = xR_list[a]
        Gi = imgs1[:,y,xL]
        x_match,cor_val= ncc_pix(Gi,y,n2, xLim, imgs2, max(xR-srch_rng,offset), max(xR+srch_rng,offset))
        pos_remove, remove_flag, entry_flag = comcor1(res2,
                                                          [x,x_match, cor_val, y], cor_thresh)
        if(remove_flag):
            res2.pop(pos_remove)
            res2.append([x,x_match, cor_val,y])
          
        elif(entry_flag):
            res2.append([x,x_match, cor_val, y])
    end = time.time()
    print('Window:')
    print(end - start) 
    print(len(res2))
    res3 = []
    start = time.time()
    for a in tqdm(range(len(xR_list))):
        xL = xL_list[a]
        y = y_list[a]
        xR = xR_list[a]
        Gi = imgs1[:,y,xL]
        x_match,cor_val= ncc_pix(Gi,y,n2, xLim, imgs2, offset, offset)
        pos_remove, remove_flag, entry_flag = comcor1(res3,
                                                          [x,x_match, cor_val, y], cor_thresh)
        if(remove_flag):
            res3.pop(pos_remove)
            res3.append([x,x_match, cor_val,y])
          
        elif(entry_flag):
            res3.append([x,x_match, cor_val,y])
    end = time.time()
    print('Reference:')
    print(end - start) 
    print(len(res3))



def disp_map2():
    #load images
    imgFolder = './test_data/testset1/bulb-multi/b1/'
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
    plt.imshow(imgs1[0])
    plt.show()
    #binary conversion
    imgs1 = biconv1(imgs1, n = n)
    
    plt.imshow(imgs1[0])
    plt.show()
    
    imgs2 = biconv3(imgs2, n = n)
    #Take left and compare to right side to find matches
    offset = 10
    rect_res = []
    xLim = imshape[1]
    yLim = imshape[0]
    for y in tqdm(range(offset, yLim-offset)):
        res_y = []
        for x in range(offset, xLim-offset):
            Gi = imgs1[:,y,x].astype('uint8')
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                x_match,cor_val= bi_pix(Gi,y,n, xLim, imgs2, offset, offset)

                pos_remove, remove_flag, entry_flag = comcor1(res_y,
                                                                  [x,x_match, cor_val, y], 0.1)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, y])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, y])
        
        rect_res.append(res_y)   
    #build disparity map
    dmap = np.zeros(imshape, dtype = 'uint8')
    dmap2 = np.zeros((imshape[0],imshape[1],3), dtype = 'uint8')
    comap = np.zeros(imshape)
    for a in rect_res:
        for b in a:
            x_c = b[0]
            y_c = b[3]
            dmap[y_c,x_c] = int(b[1] - b[0])
            comap[y_c,x_c] = b[2]
            dmap2[y_c,x_c,:] = [0,0,255]
            dmap2[y_c,b[1],:] = [255,0,0]         
    plt.imshow(dmap, cmap = 'gray')
    plt.title('DMAP')
    plt.show()

    plt.imshow(comap, cmap = 'gray')
    plt.title('COMAP')
    plt.show()
    
    plt.imshow(dmap2, cmap = 'gray')
    plt.title('DMAP2')
    plt.show()
    
    filmap = sig.medfilt2d(dmap, 3)
    plt.imshow(filmap, cmap = 'gray')
    plt.title('filMAP')
    plt.show()

def disp_map_cr():
    #load images
    imgFolder = './test_data/testset1/bulb-multi/b1/'
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
    n = imgs1.shape[0]
    #find correlations pixel accurate, without searching neighbors
    offset = 10
    rect_res = []
    xLim = imshape[1]
    yLim = imshape[0]
    #Take left and compare to right side to find matches
    for y in tqdm(range(offset, yLim-offset)):
        res_y = []
        for x in range(offset, xLim-offset):
            Gi = imgs1[:,y,x].astype('int8')
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                x_match,cor_val= ncc_pix(Gi,y,n, xLim, imgs2, offset, offset)

                pos_remove, remove_flag, entry_flag = comcor1(res_y,
                                                                  [x,x_match, cor_val, y], 0.4)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, y])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, y])
        
        rect_res.append(res_y)   
    #build disparity map
    dmap = np.zeros(imshape, dtype = 'uint8')
    
    comap = np.zeros(imshape)
    for a in rect_res:
        for b in a:
            x_c = b[0]
            y_c = b[3]
            dmap[y_c,x_c] = int(b[1] - b[0])
            comap[y_c,x_c] = b[2]
            
    plt.imshow(dmap, cmap = 'gray')
    plt.title('DMAP')
    plt.show()

    plt.imshow(comap, cmap = 'gray')
    plt.title('COMAP')
    plt.show()
    
    
    filmap = sig.medfilt2d(comap, 3)
    plt.imshow(filmap, cmap = 'gray')
    plt.title('filMAP')
    plt.show()


def test_bicos1():
    #load images
    imgFolder = './test_data/testset1/bulb/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgs1,imgs2 = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    
    
    images = 8
    n= images
    steps = 500
    stepsize = 0.01

    comp = []

    combs = list(itt.combinations(range(1, n + 1), 4))
    perm_combs = []

    for comb in combs:
        perm_combs.extend(itt.permutations(comb))

    perm_combs = np.array(sorted(perm_combs))
   # Remove unwanted permutations
    perm_combs = perm_combs[(perm_combs[:, 2] <= perm_combs[:, 3]) &
                        (perm_combs[:, 0] <= perm_combs[:, 1]) &
                        (perm_combs[:, 0] <= perm_combs[:, 2])]         
    
    bilength = perm_combs.shape[0]


    # Create matrices
    A0 = imgs1[0].astype(np.float64)
    y, x = A0.shape
    
    A_Bi = np.zeros((y, x, images))
    B_Bi = np.zeros((y, x, steps, 4), dtype=np.uint64)
    
    Am_Bi = np.zeros((y, x))
    Bm_Bi = np.zeros((y, x, steps))

    Z_map = np.zeros((y, x))
    Z_map_pixel = np.zeros((y, x))
    Bi_corr_map = np.zeros((y, x))
    Bi_Am = np.zeros((y, x))
    Liste = np.zeros((11, x * y))

    nextneighbour = 9
    half_nn = (nextneighbour - 1) // 2
    subsample_size = nextneighbour - 1
    
    sum_thresh = 300
# Create Bicos_Voxel reference sheet
    for ni in range(1):  # Load reference data
    
        B_load = np.zeros((y, x, images))
        for n in range(images):  # Load measurement data
            Bild2 = imgs1[n].astype(np.float64)
            B_load[:, :,n] = Bild2
       
        
        for x1 in tqdm(range(x)):
            for y1 in range(y):
                gB = B_load[y1, x1, :] #select pixel stack from image
                
                B_vec = np.zeros(bilength, dtype=int)#create new pixel stack for all comparison combinations
                if np.sum(gB) > sum_thresh:
                    for idx in range(bilength):
                    
                        i, j, k, l = perm_combs[idx]

                        B_vec[idx] = (gB[i - 1] + gB[j - 1]) > (gB[k - 1] + gB[l - 1])

                        if idx < 64:
                            B_Bi[y1, x1, ni, 0] = int(''.join(map(str, B_vec[:64])), 2)
                        
                        elif idx < 128:
                            B_Bi[y1, x1, ni, 1] = int(''.join(map(str, B_vec[64:128])), 2)
                        elif idx < 192:
                            B_Bi[y1, x1, ni, 2] = int(''.join(map(str, B_vec[128:192])), 2)
                        elif idx < 210:
                            B_Bi[y1, x1, ni, 3] = int(''.join(map(str, B_vec[192:210])), 2)
    print(B_Bi)



def bin_convert_test():
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
    imshape = imgs1[0].shape
    
    
    print(imshape)
    #pull a small number of images for testing
    n = 16
    comN = 4
    imgs1a = np.zeros((n,imshape[0],imshape[1]))
    imgs2a = np.zeros((n,imshape[0],imshape[1]))
    for a in range(n):
        imgs1a[a,:,:]  = imgs1[a,:,:]
        imgs2a[a,:,:] = imgs2[a,:,:]
        
        
    #load matrices
    mat_folder = './test_data/testset1/matrices/'
    kL, kR, r, t = scr.load_mats(mat_folder) 
    f = np.loadtxt(mat_folder + 'f.txt', delimiter = ' ', skiprows = 2)
    #rectify images
    v,w, H1, H2 = scr.rectify_pair(imgs1a[0], imgs2a[0], f)
    imgs1a,imgs2a = scr.rectify_lists(imgs1a,imgs2a,f)
    imgs1a = np.asarray(imgs1a)
    imgs2a = np.asarray(imgs2a)
    scr.display_stereo(v,w)
    #determine combinations of comparisons
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
    print(bilength)
    res_stack1 = np.zeros((bilength,imshape[0],imshape[1]),dtype = 'int8')
    for indval in tqdm(range(bilength)):
        i, j, k, l = perm_combs[indval]
        res_stack1[indval,:,:] = (imgs1a[i-1,:,:] + imgs1a[j-1,:,:]) > (imgs1a[k-1,:,:] + imgs1a[l-1,:,:])
    
    res_stack2 = np.zeros((bilength,imshape[0],imshape[1]),dtype = 'int8')
    for indval2 in tqdm(range(bilength)):
        i, j, k, l = perm_combs[indval2]
        res_stack2[indval2,:,:] = (imgs2a[i-1,:,:] + imgs2a[j-1,:,:]) > (imgs2a[k-1,:,:] + imgs2a[l-1,:,:])
    scr.display_stereo(res_stack1[0],res_stack2[0])
    
    indx = 500
    indy = 500
    print(res_stack1.shape)
    sdiff1 = np.sum((res_stack1[:,indy,indx] - res_stack2[:,indy,indx]))
    sdiff2 = sdiff1/bilength
    print(sdiff1)
    print(sdiff2)
    if(sdiff2 > 0):
        print(1-sdiff2)
    else:
        print(1+sdiff2)


def test_bicos2():
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
    imshape = imgs1[0].shape
    
    
    print(imshape)
    #pull a small number of images for testing
    n =8
    comN = 4
    imgs1a = np.zeros((n,imshape[0],imshape[1]))
    imgs2a = np.zeros((n,imshape[0],imshape[1]))
    for a in range(n):
        imgs1a[a,:,:]  = imgs1[a,:,:]
        imgs2a[a,:,:] = imgs2[a,:,:]
        
        
    #load matrices
    mat_folder = './test_data/testset1/matrices/'
    kL, kR, r, t = scr.load_mats(mat_folder) 
    f = np.loadtxt(mat_folder + 'f.txt', delimiter = ' ', skiprows = 2)
    #rectify images
    v,w, H1, H2 = scr.rectify_pair(imgs1[0], imgs2[0], f)
    imgs1a,imgs2a = scr.rectify_lists(imgs1,imgs2,f)
    imgs1a = np.asarray(imgs1a)
    imgs2a = np.asarray(imgs2a)
    plt.imshow(imgs1a[0])
    plt.show()
    #determine combinations of comparisons
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
    
    res_stack2 = np.zeros((bilength,imshape[0],imshape[1]),dtype = 'int8')
    for indval2 in tqdm(range(bilength)):
        i, j, k, l = perm_combs[indval2]
        res_stack2[indval2,:,:] = (imgs2a[i-1,:,:] + imgs2a[j-1,:,:]) > (imgs2a[k-1,:,:] + imgs2a[l-1,:,:])
        
    plt.imshow(res_stack1[0])
    plt.show()
    #run correlation search on stacks
    offset = 1
    rect_res = []
    xLim = imshape[1]
    yLim = imshape[0]
    #Take left and compare to right side to find matches
    for y in tqdm(range(offset, yLim-offset)):
        res_y = []
        for x in range(offset, xLim-offset):
            Gi = imgs1a[:,y,x].astype('int8')
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                x_match,cor_val,subpix = ncc.cor_acc_pix(Gi,y,n, xLim, imgs2a, offset, offset)

                pos_remove, remove_flag, entry_flag = ncc.compare_cor(res_y,
                                                                  [x,x_match, cor_val, subpix, y], 0.9)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, subpix, y])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, subpix, y])
        
        rect_res.append(res_y)
    '''
    #Compare the found right side and compare to left side, and see if it matches. If not, discard.
    rect_res2 = []
    for entvb in rect_res:
        ent2 = []
        for ent in entvb:
            Gi2 = imgs2a[:,int(ent[4]),int(ent[0])]
            x_match2, cor_val2, subpix2 = ncc.cor_acc_pix(Gi2,ent[4],n,xLim,imgs1a,offset,offset)
            if(x_match2+subpix2[1] == ent[1]+ent[3][1]):
                ent2.append(ent)
        rect_res2.append(ent2)
            
    '''
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
            
             
    col_ptsL = np.around(ptsL,0).astype('uint16')
    col_ptsR = np.around(ptsR,0).astype('uint16')
    
    col_arr = scr.get_color(col_refL, col_refR, col_ptsL, col_ptsR)
    tri_res = scr.triangulate_list(ptsL,ptsR, r, t, kL, kR)
    scr.convert_np_ply(np.asarray(tri_res), col_arr,'test_bicos.ply')


def bicos_cor_clean(res_list, entry_val, threshold):
    remove_flag = False
    pos_remove = 0
    entry_flag = False
    counter = 0
    if(entry_val[1] < 0 or entry_val[2] < threshold):
        return pos_remove,remove_flag,entry_flag
    for i in range(len(res_list)):       
        
        if(res_list[i][1] == entry_val[1] and res_list[i][3][0] - entry_val[3][0] < float_epsilon and
           res_list[i][3][1] - entry_val[3][1] < float_epsilon):
            #duplicate found, check correlation values and mark index of previous entry for removal if new entry has higher correlation 
            remove_flag = (res_list[i][2] < entry_val[2])
            pos_remove = i
            break
        else:
            counter+=1
    #end of list reached, no duplicates found, entry is valid
    if(counter == len(res_list)):
        entry_flag = True
    return pos_remove,remove_flag,entry_flag
@numba.jit(nopython=True)
def bicos_comp(Gi,y,n, xLim, maskR, xOffset1, xOffset2):
    max_cor = 0 
    max_index = -1
    max_mod = [0,0] #default to no change
    #Search the entire line    
    for xi in range(xOffset1, xLim-xOffset2):
        Gt = maskR[:,y,xi]
        chkres = Gi-Gt
        chk = np.count_nonzero(chkres == 0)
        if(chk > max_cor):
            max_index = xi
            max_cor = chk
    #search surroundings of found best match
    Gs = maskR[:,y-1, max_index]
    chkres = Gi-Gs
    chk = np.count_nonzero(chkres == 0)
    if(chk > max_cor):
        max_index = xi
        max_cor = chk
        max_mod = [-1,0]
    Gs = maskR[:,y-1, max_index-1]
    chkres = Gi-Gs
    chk = np.count_nonzero(chkres == 0)
    if(chk > max_cor):
        max_index = xi
        max_cor = chk
        max_mod = [-1,-1]
        
    Gs = maskR[:,y-1, max_index+1]
    chkres = Gi-Gs
    chk = np.count_nonzero(chkres == 0)
    if(chk > max_cor):
        max_index = xi
        max_cor = chk
        max_mod = [-1,1]
    
    Gs = maskR[:,y-1, max_index]
    chkres = Gi-Gs
    chk = np.count_nonzero(chkres == 0)
    if(chk > max_cor):
        max_index = xi
        max_cor = chk
        max_mod = [1,0]
        
    
    Gs = maskR[:,y+1, max_index-1]
    chkres = Gi-Gs
    chk = np.count_nonzero(chkres == 0)
    if(chk > max_cor):
        max_index = xi
        max_cor = chk
        max_mod = [1,-1]
        
    Gs = maskR[:,y+1, max_index+1]
    chkres = Gi-Gs
    chk = np.count_nonzero(chkres == 0)
    if(chk > max_cor):
        max_index = xi
        max_cor = chk
        max_mod = [1,1]
        
    return max_index,max_cor,max_mod
    
def test_bicos3():
    #load images
    imgFolder = './test_data/testset1/bulb-multi/b1/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgs1,imgs2 = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    col_refL, col_refR = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd,colorIm = True)
    
    #apply filter
    thresh1 = 30
    imgs1 = np.asarray(scr.mask_inten_list(imgs1,thresh1))
    imgs2 = np.asarray(scr.mask_inten_list(imgs2,thresh1))
    imshape = imgs1[0].shape
    
    
    print(imshape)
    #load matrices
    mat_folder = './test_data/testset1/matrices/'
    kL, kR, r, t = scr.load_mats(mat_folder) 
    f = np.loadtxt(mat_folder + 'f.txt', delimiter = ' ', skiprows = 2)
    
    
    
    #rectify images
    v,w, H1, H2 = scr.rectify_pair(imgs1[0], imgs2[0], f)
    imgs1,imgs2 = scr.rectify_lists(imgs1,imgs2,f)
    imgs1 = np.asarray(imgs1)
    imgs2 = np.asarray(imgs2)
    
    
    #Generate binary pixel stacks
    comN = 4
    n = 12
    
    imgs1a = np.zeros((n,imshape[0],imshape[1]))
    imgs2a = np.zeros((n,imshape[0],imshape[1]))
    for a in range(n):
        imgs1a[a,:,:]  = imgs1[a,:,:]
        imgs2a[a,:,:] = imgs2[a,:,:]
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
    
    res_stack2 = np.zeros((bilength,imshape[0],imshape[1]),dtype = 'int8')
    for indval2 in tqdm(range(bilength)):
        i, j, k, l = perm_combs[indval2]
        res_stack2[indval2,:,:] = (imgs2a[i-1,:,:] + imgs2a[j-1,:,:]) > (imgs2a[k-1,:,:] + imgs2a[l-1,:,:])
    #compare features
    offset = 1
    rect_res = []
    xLim = imshape[1]
    yLim = imshape[0]
    #Take left and compare to right side to find matches
    for y in tqdm(range(offset, yLim-offset)):
        res_y = []
        for x in range(offset, xLim-offset):
            Gi = imgs1a[:,y,x].astype('int8')
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices
                x_match,cor_val,subpix = bicos_comp(Gi,y,n, xLim, imgs2a, offset, offset)

                pos_remove, remove_flag, entry_flag = bicos_cor_clean(res_y,
                                                                  [x,x_match, cor_val, subpix, y],n-n/10)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, subpix, y])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, subpix, y])
        
        rect_res.append(res_y)
    #create disparity map
    
    #apply median filter to disparity map
    
    #unrectify points and triangulate
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
            
             
    col_ptsL = np.around(ptsL,0).astype('uint16')
    col_ptsR = np.around(ptsR,0).astype('uint16')
    
    col_arr = scr.get_color(col_refL, col_refR, col_ptsL, col_ptsR)
    tri_res = scr.triangulate_list(ptsL,ptsR, r, t, kL, kR)
    scr.convert_np_ply(np.asarray(tri_res), col_arr,'test_bicos.ply')
    



def unpack_rect_res(listin):
    pts1 = []
    pts2 = []
    cor = []
    
    #[x,x_match, cor_val, subpix, y]
    for i in listin:
        for j in i:
            pts1.append([j[4] + j[3][0],j[0]+j[3][1]])
            pts2.append([j[4]+ j[3][0],j[1]]+j[3][1])
            cor.append(j[2])
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    cor = np.array(cor)
    
    pts1int = pts1.astype("uint32")
    pts2int = pts2.astype("uint32")
    
    return pts1,pts2,pts1int,pts2int,cor
    
        
def sp_mond():
    #load images
    imgFolder = './test_data/testset0/moon2/'
    imgs = scr.load_all_imgs_1_dir(imgFolder, convert_gray = True)
    img1 = imgs[0]
    img2 = imgs[1]

    thresh = 10
    img1 = scr.mask_img(img1,thresh)
  
    img2 = scr.mask_img(img2,thresh)
    scr.display_stereo(img1,img2)
    img1 = scr.boost_zone(img1,1.5,1,1,1,1)
    img2 = scr.boost_zone(img2,1.5,1,1,1,1)
    scr.display_stereo(img1,img2)
    
    t1 = spat_extract(img1)
    t2 = spat_extract(img2)
    
    n = 49
    offset = 50
    imshape = img1.shape
    rect_res = []
    for y in tqdm(range(offset, imshape[0]-offset)):
        res_y = []
        for x in range(offset, imshape[1]-offset):
            Gi = t1[:,y,x]
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices

                x_match,cor_val,subpix = ncc.cor_acc_rbf(Gi,y,n, imshape[1], t2, offset, offset, 3)


                
                pos_remove, remove_flag, entry_flag = ncc.compare_cor(res_y,[x,x_match, cor_val, subpix, y], 0.9)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, subpix, y])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, subpix, y])
        
        rect_res.append(res_y)
        
    pts1,pts2,pts1int,pts2int,cor = unpack_rect_res(rect_res)
    scr.mark_points(img1,img2, pts1int,pts2int, size = 1)
    

def test_sp1():
    #load images
    imgFolder = './test_data/testset1/bulb/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgs1,imgs2 = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    test1 = spat_extract(imgs1[0])
    test2 = spat_extract(imgs2[0])
    #load data text file
    filename = "bulbcorr.txt"
    
    pts1,pts2,geom_arr,col_arr,correl = scr.read_txt(filename)
    print(test1.shape)
    print(imgs1.shape)
    indchk = 0
    chk1 = test1[:,int(pts1[indchk,0]),int(pts1[indchk,1])]
    chk2 = test2[:,int(pts2[indchk,0]),int(pts2[indchk,1])]
    print(pts1[indchk])
    print(pts2[indchk])
    
    agi1 = np.sum(chk1)/test1.shape[0] 
    val_i1 = np.sum((chk1-agi1)**2)
    agt1 = np.sum(chk2)/test1.shape[0]        
    val_t1 = np.sum((chk2-agt1)**2)
    cor1 = np.sum((chk1-agi1)*(chk2 - agt1))/(np.sqrt(val_i1*val_t1))
    print(cor1)
    

    print(correl[indchk])


def test_sp2():
    #load images
    imgFolder = './test_data/testset1/bulb/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgs1,imgs2 = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    #load matrices
    matFolder = './test_data/testset1/matrices/'
    f_file = 'f.txt'
    kL, kR, r, t = scr.load_mats(matFolder)
    F = np.loadtxt(matFolder + f_file, delimiter = ' ', skiprows = 2)
    #load image stacks
    test1 = spat_extract(imgs1[0])
    test2 = spat_extract(imgs2[0])
    #rectify image stacks
    rectL,rectR = scr.rectify_lists(test1,test2, F)
    avgL = np.asarray(rectL).mean(axis=(0))
    avgR = np.asarray(rectR).mean(axis=(0))

    #Background filter
    thresh_val = 1
    maskL = scr.mask_avg_list(avgL,rectL, thresh_val)
    maskR = scr.mask_avg_list(avgR,rectR, thresh_val)
    n = 49
    maskL = np.asarray(maskL)
    maskR = np.asarray(maskR)
    #run ncc on stacks
    offset = 10
    imshape = imgs1[0].shape
    rect_res = []
    for y in tqdm(range(offset, imshape[0]-offset)):
        res_y = []
        for x in range(offset, imshape[1]-offset):
            Gi = maskL[:,y,x]
            if(np.sum(Gi) > float_epsilon): #dont match fully dark slices

                x_match,cor_val,subpix = ncc.cor_acc_pix(Gi,y,n, imshape[1], maskR, offset, offset)


                
                pos_remove, remove_flag, entry_flag = ncc.compare_cor(res_y,[x,x_match, cor_val, subpix, y], 0.9)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, subpix, y])
                  
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, subpix, y])
        
        rect_res.append(res_y)

    im_a,im_b,HL,HR = scr.rectify_pair(test1[0],test2[0], F)
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

    col_arr = scr.gen_color_arr_black(len(ptsL))
    

    tri_res = scr.triangulate_list(ptsL,ptsR, r, t, kL, kR)
    
    scr.convert_np_ply(np.asarray(tri_res), col_arr,' .ply')


def calc_f_pts():

    #load data text file
    filename = "bulbcorr.txt"
    
    pts1,pts2,geom_arr,col_arr,correl = scr.read_txt(filename)
    
    #feed them into calc F function
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)
    #check with reference F matrix
    f_file = "./test_data/testset1/matrices/f.txt"
    f_mat = np.loadtxt(f_file, delimiter = " ", skiprows =2)
    print("REF F:")
    print(f_mat)
    print("TEST F:")
    print(F)
    
   


def check_ncc_fmat():
    #load image stacks
    imgFolder = './test_data/testset1/bulb/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgsL,imgsR = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    #use calc fmat ncc
    f_mat = scr.find_f_mat_ncc(imgsL, imgsR, thresh = 0.9)
    #compare with reference f matrix
    f_file = "./test_data/testset1/matrices/f.txt"
    F = np.loadtxt(f_file, delimiter = " ", skiprows =2)
    print("REF F:")
    print(F)
    print("TEST F:")
    print(f_mat)



    
def check_list_fmat():
    imgFolder = './test_data/testset1/bulb/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgsL,imgsR = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    f_mat = scr.find_f_mat_list(imgsL,imgsR,thresh = 0.9)
    f_file = "./test_data/testset1/matrices/f.txt"
    F = np.loadtxt(f_file, delimiter = " ", skiprows =2)
    print("REF F:")
    print(F)
    print("TEST F:")
    print(f_mat)



    
def test_contrast_boost():
    imgFolder1 = './test_data/testset2/020000us/'
    
    imgFolder = './test_data/testset2/000500us/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgsL,imgsR = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    imgsL1,imgsR1 = scr.load_images_1_dir(imgFolder1, imgLInd, imgRInd)
    scr.display_4_comp(imgsL[1], imgsR[1], imgsL1[1], imgsR1[1])
    
    
    factor = 50
    lef =scr.boost_zone(imgsL[1],factor, 1, 1, 1, 1)
    ri =scr.boost_zone(imgsR[1], factor, 1, 1, 1, 1)
    scr.display_4_comp(imgsL[1],imgsR[1],lef, ri)


def test_corr_cal():
    
    #Inputs: kL, kR, right and left images
    #Outputs: F, R, t
    mat_folder = './test_data/testset1/matrices/'
    #load kL, kR
    kL_file = mat_folder + 'kL.txt'
    kR_file = mat_folder + 'kR.txt'
    kL = np.loadtxt(kL_file, skiprows=2, delimiter = ' ')
    kR= np.loadtxt(kR_file, skiprows=2, delimiter = ' ')

    #load known R,t,F values
    R_file = mat_folder + 'R.txt'
    t_file = mat_folder + 't.txt'
    f_file = mat_folder + 'f.txt'
    R = np.loadtxt(R_file, skiprows=2, delimiter = ' ')
    t= np.loadtxt(t_file, skiprows=2, delimiter = ' ')
    f = np.loadtxt(f_file, skiprows=2, delimiter = ' ')

    #Load images
    imgFolder = './test_data/testset1/bulb/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgsL,imgsR = scr.load_images_1_dir(imgFolder, imgLInd, imgRInd)
    
    #compute F matrix from images, and save points used in process
    f_test, pts1_rec,pts2_rec = scr.find_f_mat_list(imgsL,imgsR, thresh = 0.9, f_calc_mode = 0, ret_pts = True)   
    #f_test, pts1_rec,pts2_rec = scr.find_f_mat_ncc(imgsL, imgsR, thresh = 0.9, ret_pts = True)
    
    #apply corr_cal to get R, t
    #R_test, t_test = scr.corr_calibrate(pts1_rec,pts2_rec, kL, kR, f_test)
    
    ess = kR.T @ f_test @ kL
    #a,R_test,t_test,b = cv2.recoverPose(ess,pts1_rec,pts2_rec)
    
    #t_test=t_test.T[0]
    
    s = cv2.decomposeEssentialMat(ess)
    print(s)
    ber = scr.triangulate(pts1_rec[0],pts2_rec[0], R, t, kL, kR)
    a = scr.triangulate(pts1_rec[0],pts2_rec[0], s[0], s[2], kL, kR)
    b = scr.triangulate(pts1_rec[0],pts2_rec[0], s[1], s[2], kL, kR)
    c = scr.triangulate(pts1_rec[0],pts2_rec[0], s[0], -s[2], kL, kR)
    d = scr.triangulate(pts1_rec[0],pts2_rec[0], s[1], -s[2], kL, kR)
    print('Ref R, Ref t')
    print(ber)
    print('R1,t')
    print(a)
    print('R2,t')
    print(b)
    print('R1,-t')
    print(c)
    print('R2,-t')
    print(d)
    
    if a[2] > 0:
        R_test = s[0]
    else:
        R_test = s[1]
        
    t_test = s[2].T[0]
    
    a = scr.triangulate(pts1_rec[0],pts2_rec[0], s[0], s[2]*np.linalg.norm(t), kL, kR)
    b = scr.triangulate(pts1_rec[0],pts2_rec[0], s[1], s[2]*np.linalg.norm(t), kL, kR)
    c = scr.triangulate(pts1_rec[0],pts2_rec[0], s[0], -s[2]*np.linalg.norm(t), kL, kR)
    d = scr.triangulate(pts1_rec[0],pts2_rec[0], s[1], -s[2]*np.linalg.norm(t), kL, kR)
    print('R1,t scaled')
    print(a)
    print('R2,t scaled') 
    print(b)
    print('R1,-t scaled')
    print(c)
    print('R2,-t scaled')
    print(d)
    
    #compare matrices
    print('Ref f:')
    print(f)
    print('Test f:')
    print(f_test)
    print('Ref R:')
    print(R)
    print('Test R:')
    print(R_test)
    print('ess R')
    print(R_test)
    print('rpose R')
    m,n,k,l = cv2.recoverPose(ess,pts1_rec, pts2_rec)
    print(n)    
    #compare t magnitude
    print("T Mag:")
    print(np.linalg.norm(t))
    print("Test T Mag:")
    print(np.linalg.norm(t_test))   
    print('Ref t:')
    print(t)
    print('Test t:')
    print(t_test)
    print("Test t * T mag:")
    print(t_test * np.linalg.norm(t))
    
    #test rectification with found f matrix
    img1,img2, H1, H2 = scr.rectify_pair(imgsL[0],imgsR[0], f)
    img1_test,img2_test, H1, H2 = scr.rectify_pair(imgsL[0],imgsR[0], f_test)
    
    scr.display_4_comp(img1,img2,img1_test,img2_test)



def create_diff_grid(img_stk, thresh = 5):
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



def test_diff_grid():
    #load images
    img_folder = './test_data/testset0/240312_angel/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imagesL,imagesR = scr.load_images_1_dir(img_folder, imgLInd, imgRInd)
    #rectify images
    f_mat = np.loadtxt("./test_data/testset0/matrices/f.txt", skiprows=2, delimiter = " ")
    rectL, rectR = scr.rectify_lists(imagesL,imagesR, f_mat)
    #convert to diff grids
    thr = 10
    diffL = create_diff_grid(rectL,thresh = thr)
    diffR = create_diff_grid(rectR, thresh = thr)
    n = diffL.shape[0]
    posX = 700
    posY = 600
    max_cor = 0.0
    max_index = -1
    Gi = diffL[:,posY,posX].astype('int16')
    print(Gi)
    print('###########################')
    offset = 100
    interval = 50
    for xi in range(offset,diffL[0].shape[1]-offset,interval):
 
        Gt = diffR[:,posY,xi].astype('int16')
        vc = np.sum((Gi-Gt)**2)/n
        
        if vc > 0:
            cor = 1-vc
        else:
            cor = 1+vc
        print(Gt)
        print((Gi-Gt)**2)
        if cor > max_cor:
            max_cor = cor
            max_index = xi
        print(cor)
    Gt = diffR[:,posY,max_index]
    print('###########################')
    print(Gt)
    print(max_cor)

@numba.jit(nopython=True)   
def col_help(lims, imagesL, i, thresh, res_red, res_red_count, res_green, res_green_count, res_blue, res_blue_count):
    for j in range(lims[1]):
        val_stack = imagesL[:,i,j,:]
        for a in range(val_stack.shape[0]):
            r_val = val_stack[a,0]
            if r_val > thresh:
                res_red[i,j] += r_val
                res_red_count[i,j] += 1
            g_val = val_stack[a,1]
            if g_val > thresh:
                res_green[i,j] += g_val
                res_green_count[i,j] += 1
            b_val = val_stack[a,2]
            if b_val > thresh:
                res_blue[i,j] += b_val
                res_blue_count[i,j] += 1
    return res_red, res_green, res_blue, res_red_count, res_green_count, res_blue_count

def col_ex():
    #recreates un-patterned image in color from stack of structured illumination applied images
    img_folder = './test_data/testset0/240312_angel/'
    #img_folder = './test_data/testset0/240411_hand0/'
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imagesL,imagesR = scr.load_images_1_dir(img_folder, imgLInd, imgRInd, colorIm = True)
    #create 7 empty arrays of same shape as image, 3 to store running sums of each channel, 3 to store count of values added, 1 for result
    res_image = np.zeros(imagesL[0].shape)
    res_red = np.zeros(imagesL[0,:,:,0].shape)
    res_red_count = np.ones(imagesL[0,:,:,0].shape)
    res_blue = np.zeros(imagesL[0,:,:,0].shape)
    res_blue_count = np.ones(imagesL[0,:,:,0].shape)
    res_green = np.zeros(imagesL[0,:,:,0].shape)
    res_green_count = np.ones(imagesL[0,:,:,0].shape)
    #establish color intensity thresholds for rejection of value
    thresh = 10
    lims = imagesL[0].shape
    #loop through stack of images 3xn and retrieve all 3 color channels for each pixel for each image
    for i in range(lims[0]):
        res_red, res_green, res_blue, res_red_count, res_green_count, res_blue_count = col_help(lims, imagesL, i, thresh, res_red, res_red_count, res_green, res_green_count, res_blue, res_blue_count)
    res_image[:,:,0] = res_red/res_red_count/255
    res_image[:,:,1] = res_green/res_green_count/255
    res_image[:,:,2] = res_blue/res_blue_count/255
      
    plt.imshow(imagesL[0])
    plt.show()
    plt.imshow(res_image)
    plt.show()   
    
def get_color1(imagesL,ptsL, mode = 1):
    #create 7 empty arrays of same shape as image, 3 to store running sums of each channel, 3 to store count of values added, 1 for result
    res_imageL = np.zeros(imagesL[0].shape)
    res_redL = np.zeros(imagesL[0,:,:,0].shape)
    res_red_countL = np.ones(imagesL[0,:,:,0].shape)
    res_blueL = np.zeros(imagesL[0,:,:,0].shape)
    res_blue_countL = np.ones(imagesL[0,:,:,0].shape)
    res_greenL = np.zeros(imagesL[0,:,:,0].shape)
    res_green_countL = np.ones(imagesL[0,:,:,0].shape)
    

    #establish color intensity thresholds for rejection of value
    thresh = 10
    lims = imagesL[0].shape
    #loop through stack of images 3xn and retrieve all 3 color channels for each pixel for each image
    for i in range(lims[0]):
        res_redL, res_greenL, res_blueL, res_red_countL, res_green_countL, res_blue_countL = col_help(lims, imagesL, i, thresh, res_redL, res_red_countL, res_greenL, res_green_countL, res_blueL, res_blue_countL)
       
    res_imageL[:,:,0] = res_redL/res_red_countL/255
    res_imageL[:,:,1] = res_greenL/res_green_countL/255
    res_imageL[:,:,2] = res_blueL/res_blue_countL/255 
    

    
    res_col = []
    blcn = 0
    for a in range(len(ptsL)):
        try:
            if(mode == 0):
                res_col.append((res_imageL[ptsL[a][0],ptsL[a][1],:]))
            else:
                res_col.append((res_imageL[ptsL[a][1],ptsL[a][0],:]))
        except:
            res_col.append(np.asarray([0,0,0]))
            blcn+= 1
    res_col = np.asarray(res_col)
    return res_col, blcn

def get_color2(imagesL,imagesR,ptsL,ptsR, mode = 1):
    #create 7 empty arrays of same shape as image, 3 to store running sums of each channel, 3 to store count of values added, 1 for result
    res_imageL = np.zeros(imagesL[0].shape)
    res_redL = np.zeros(imagesL[0,:,:,0].shape)
    res_red_countL = np.ones(imagesL[0,:,:,0].shape)
    res_blueL = np.zeros(imagesL[0,:,:,0].shape)
    res_blue_countL = np.ones(imagesL[0,:,:,0].shape)
    res_greenL = np.zeros(imagesL[0,:,:,0].shape)
    res_green_countL = np.ones(imagesL[0,:,:,0].shape)
    
    res_imageR = np.zeros(imagesR[0].shape)
    res_redR = np.zeros(imagesR[0,:,:,0].shape)
    res_red_countR = np.ones(imagesR[0,:,:,0].shape)
    res_blueR = np.zeros(imagesR[0,:,:,0].shape)
    res_blue_countR = np.ones(imagesR[0,:,:,0].shape)
    res_greenR = np.zeros(imagesR[0,:,:,0].shape)
    res_green_countR = np.ones(imagesR[0,:,:,0].shape)
    #establish color intensity thresholds for rejection of value
    thresh = 10
    lims = imagesL[0].shape
    #loop through stack of images 3xn and retrieve all 3 color channels for each pixel for each image
    for i in range(lims[0]):
        res_redL, res_greenL, res_blueL, res_red_countL, res_green_countL, res_blue_countL = col_help(lims, imagesL, i, thresh, res_redL, res_red_countL, res_greenL, res_green_countL, res_blueL, res_blue_countL)
        res_redR, res_greenR, res_blueR, res_red_countR, res_green_countR, res_blue_countR = col_help(lims, imagesR, i, thresh, res_redR, res_red_countR, res_greenR, res_green_countR, res_blueR, res_blue_countR)
    res_imageL[:,:,0] = res_redL/res_red_countL/255
    res_imageL[:,:,1] = res_greenL/res_green_countL/255
    res_imageL[:,:,2] = res_blueL/res_blue_countL/255 
    
    res_imageR[:,:,0] = res_redR/res_red_countR/255
    res_imageR[:,:,1] = res_greenR/res_green_countR/255
    res_imageR[:,:,2] = res_blueR/res_blue_countR/255 

    
    res_col = []
    blcn = 0
    for a in range(len(ptsL)):
        try:
            if(mode == 0):
                res_col.append((res_imageL[ptsL[a][0],ptsL[a][1],:]))
            else:
                res_col.append((res_imageL[ptsL[a][1],ptsL[a][0],:]))
        except:
            try:
                if(mode == 0):
                    res_col.append((res_imageR[ptsR[a][0],ptsR[a][1],:]))
                else:
                    res_col.append((res_imageR[ptsR[a][1],ptsR[a][0],:]))
            except:
                res_col.append(np.asarray([0,0,0]))
                blcn+= 1
    res_col = np.asarray(res_col)
    return res_col, blcn


def test_distort_comp():
    #load images
    folder = './test_data/testset0/240312_angel/'
    
    imgL,imgR = scr.load_images_1_dir(folder, 'cam1', 'cam2', ext = '.jpg', colorIm= True)
    #generate test distortion compensation vectors
    distL = np.asarray([0, 0, 1, 1, 0])
    distR = np.asarray([0, 0, 1, 1, 0])
    #load camera matrices
    mat_folder = './test_data/testset0/matrices/'
    kL, kR, R, t = scr.load_mats(mat_folder)
    #apply distortion compensators to images
    img_nL = imgL[0]
    img_nR = imgR[0]
    h, w = img_nL.shape[:2]
    new_kL, roiL = cv2.getOptimalNewCameraMatrix(kL, distL, (w,h), 1, (w,h))
    new_kR, roiR = cv2.getOptimalNewCameraMatrix(kR, distR, (w,h), 1, (w,h))
    dstL = cv2.undistort(img_nL, kL, distL, None, new_kL)
    dstR = cv2.undistort(img_nR, kR, distR, None, new_kR)
    #display test image set
    plt.imshow(imgL[0])
    plt.show()
    plt.imshow(dstL)
    plt.show()
    plt.imshow(imgR[0])
    plt.show()
    plt.imshow(dstR)
    plt.show()
    print(kL)
    print(new_kL)

def test_col_recon():
    #Load images
    ref_file = '000POS000Rekonstruktion030.pcf'
    folder = './test_data/testset0/240312_angel/'

    imgL,imgR = scr.load_images_1_dir(folder, 'cam1', 'cam2', ext = '.jpg', colorIm= True)
    #Load pcf
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(folder + ref_file)
    #get color points from images and pcf points
    #truncate points to integers
    ptsL = np.around(xy1,0).astype('uint16')
    col_arr2 = scr.get_color(imgL, ptsL)
    #create ply file
    scr.convert_np_ply(geom_arr, col_arr2, 'test_color.ply')
    
    


def comp_color_source():
    #load images
    #get correlated points
    config = chand.ConfigHandler()
    config.mat_folder = './test_data/testset0/matrices/'
    config.sing_img_folder = './test_data/testset0/240312_angel/'
    config.f_mat_file_mode = 1
    ptsL,ptsR = ncc.cor_pts(config)
    imagesL,imagesR = scr.load_images_1_dir(config.sing_img_folder, config.sing_left_ind, config.sing_right_ind, config.sing_ext, colorIm = True)
    col_ptsL = np.around(ptsL,0).astype('uint16')
    col_ptsR = np.around(ptsR,0).astype('uint16')
    res_col1, blcn1 = get_color1(imagesL, col_ptsL, mode = 0)
    res_col2, blcn2 = get_color2(imagesL,imagesR, col_ptsL, col_ptsR, mode = 0)
    print(blcn1)
    print(blcn2)
    print("######")
    print(len(res_col1))



   
def test_warp():
    #load images
    img_folder = './test_data/testset0/240312_boat/'
    left_ind = "cam1"
    right_ind = "cam2"
    ext = ".jpg"
    imgL,imgR = scr.load_images_1_dir(img_folder, left_ind, right_ind, ext)
    #rectify images
    f_mat = np.loadtxt("./test_data/testset0/matrices/f.txt", skiprows=2, delimiter = " ")
    rectL, rectR, H1, H2 = scr.rectify_pair(imgL[0],imgR[0], f_mat)
    #apply inverse matrix to rectified image
    chkL = cv2.warpPerspective(rectL, np.linalg.inv(H1), [rectL.shape[1],rectL.shape[0]])
    #check results
    plt.imshow(imgL[0])
    plt.show()
    plt.imshow(rectL)
    plt.show()
    plt.imshow(chkL)
    plt.show()

def test_warp2():
    #load images
    img_folder = './test_data/testset0/240312_fruit/'
    left_ind = "cam1"
    right_ind = "cam2"
    ext = ".jpg"
    imagesL,imagesR = scr.load_images_1_dir(img_folder, left_ind, right_ind, ext)
    imagesL_col,imagesR_col = scr.load_images_1_dir(img_folder, left_ind, right_ind, ext, colorIm = True)
    #load matrices
    mat_folder = './test_data/testset0/matrices/'
    kL, kR, R, t = scr.load_mats(mat_folder)
    f_file = 'f.txt'
    f_mat = np.loadtxt(mat_folder + f_file, skiprows=2, delimiter = ' ')
    #rectify images
    rectL,rectR = scr.rectify_lists(imagesL,imagesR, f_mat)
    avgL = np.asarray(rectL).mean(axis=(0))
    avgR = np.asarray(rectR).mean(axis=(0))

    #Background filter
    thresh_val = 10
    maskL = scr.mask_avg_list(avgL,rectL, thresh_val)
    maskR = scr.mask_avg_list(avgR,rectR, thresh_val)

    maskL = np.asarray(maskL)
    maskR = np.asarray(maskR)
    imshape = imagesL[0].shape
    #correlate points on images
    xLim = imshape[1]
    yLim = imshape[0]
    xOffsetL = 1
    xOffsetR = 1
    yOffsetT = 1
    yOffsetB = 1
    thresh = 0.9
    interp = 3
    rect_res = []
    n = len(imagesL)
    interval = 1
    for y in tqdm(range(yOffsetT, yLim-yOffsetB)):
        res_y = []
        for x in range(xOffsetL, xLim-xOffsetR, interval):
            Gi = maskL[:,y,x]
            if(np.sum(Gi) != 0): #dont match fully dark slices
                x_match,cor_val,subpix = ncc.cor_acc_rbf(Gi,y,n, xLim, maskR, xOffsetL, xOffsetR, interp)
                    
                pos_remove, remove_flag, entry_flag = ncc.compare_cor(res_y,
                                                                  [x,x_match, cor_val, subpix, y], thresh)
                if(remove_flag):
                    res_y.pop(pos_remove)
                    res_y.append([x,x_match, cor_val, subpix, y])
                elif(entry_flag):
                    res_y.append([x,x_match, cor_val, subpix, y])
        rect_res.append(res_y)      
    #unrectify points using old method
    im_a,im_b,HL,HR = scr.rectify_pair(imagesL[0],imagesR[0], f_mat)
    hL_inv = np.linalg.inv(HL)
    hR_inv = np.linalg.inv(HR)
    ptsL = []
    ptsR = []
    for a in range(len(rect_res)):
        b = rect_res[a]
        for q in b:
            sL = HL[2,0]*q[0] + HL[2,1] * (q[4]) + HL[2,2]
            pL = hL_inv @ np.asarray([[q[0]],[q[4]],[sL]])
            sR = HR[2,0]*(q[1] + q[3][1]) + HR[2,1] * (q[4]+q[3][0]) + HR[2,2]
            pR = hR_inv @ np.asarray([[q[1]+ q[3][1]],[q[4]+q[3][0]],[sR]])
            ptsL.append([pL[0,0],pL[1,0],pL[2,0]])
            ptsR.append([pR[0,0],pR[1,0],pR[2,0]])
            
            
    #check valid points
    ptsL = scr.conv_pts(ptsL)
    ptsR = scr.conv_pts(ptsR)
    print(ptsL[0])
    print(ptsR[0])
    print(maskL[0].shape)
    print("########")
    print("xLim: " + str(xLim))
    print("yLim:" + str(yLim))
    print("######")
    #check number of valid unrectified points
    invalid_Lx = 0
    invalid_Ly = 0
    invalid_Rx = 0
    invalid_Ry = 0
    for a,b in zip(ptsL,ptsR):
        if a[1] > xLim or a[1] < 0:
            invalid_Lx += 1
        elif a[0] > yLim or a[0] < 0:
            invalid_Ly += 1
        if b[1] > xLim or b[1] < 0:
            invalid_Rx += 1
        elif b[0] > yLim or b[0] < 0:
            invalid_Ry += 1
    print("Invalid Left X-vals:")
    print(invalid_Lx)
    print("Invalid Left Y-vals:")
    print(invalid_Ly)
    print("Invalid Right X-vals:")
    print(invalid_Rx)
    print("Invalid Right Y-vals:")
    print(invalid_Ry) 
    #unrectify points with new method
    ptsLn = []
    ptsRn = []
    for c in range(len(rect_res)):
        d = rect_res[c]
        for g in d:
            xL = g[0]
            y = g[4]
            xR = g[1]
            xL_u = (hL_inv[0,0]*xL + hL_inv[0,1] * y + hL_inv[0,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * y + hL_inv[2,2])
            yL_u = (hL_inv[1,0]*xL + hL_inv[1,1] * y + hL_inv[1,2])/(hL_inv[2,0]*xL + hL_inv[2,1] * y + hL_inv[2,2])
            xR_u = (hR_inv[0,0]*xR + hR_inv[0,1] * y + hR_inv[0,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * y + hR_inv[2,2])
            yR_u = (hR_inv[1,0]*xR + hR_inv[1,1] * y + hR_inv[1,2])/(hR_inv[2,0]*xL + hR_inv[2,1] * y + hR_inv[2,2])
            ptsLn.append([xL_u,yL_u])
            ptsRn.append([xR_u,yR_u])
    
    #check number of valid unrectified points
    print(ptsLn[0])
    print(ptsRn[0])
    invalid_Lx = 0
    invalid_Ly = 0
    invalid_Rx = 0
    invalid_Ry = 0
    for a,b in zip(ptsLn,ptsRn):
        if a[1] > xLim or a[1] < 0:
            invalid_Lx += 1
        elif a[0] > yLim or a[0] < 0:
            invalid_Ly += 1
        if b[1] > xLim or b[1] < 0:
            invalid_Rx += 1
        elif b[0] > yLim or b[0] < 0:
            invalid_Ry += 1
    print('####################')
    print("Invalid Left X-vals:")
    print(invalid_Lx)
    print("Invalid Left Y-vals:")
    print(invalid_Ly)
    print("Invalid Right X-vals:")
    print(invalid_Rx)
    print("Invalid Right Y-vals:")
    print(invalid_Ry) 
    tri_res = scr.triangulate_list(ptsLn,ptsRn, R, t, kL, kR)
    col_pts = np.around(ptsLn,0).astype('uint16')
    
    col_arr = scr.get_color(imagesL_col,col_pts,1)
    scr.convert_np_ply(np.asarray(tri_res), col_arr,'new_rect.ply')
    
    
def check_ref_valid():
    #load images
    ref_file = '000POS000Rekonstruktion030.pcf'
    folder = './test_data/testset0/240312_fruit/'
    imgL,imgR = scr.load_images_1_dir(folder, 'cam1', 'cam2', ext = '.jpg', colorIm = True)
    #load pcf
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(folder + ref_file)
    xLim = imgL[0].shape[1] - 1
    yLim = imgL[0].shape[0] - 1
    invalid_Lx = 0
    invalid_Ly = 0
    invalid_Rx = 0
    invalid_Ry = 0
    for a,b in zip(xy1,xy2):
        if a[1] > xLim or a[1] < 0:
            invalid_Lx += 1
        elif a[0] > yLim or a[0] < 0:
            invalid_Ly += 1
        if b[1] > xLim or b[1] < 0:
            invalid_Rx += 1
        elif b[0] > yLim or b[0] < 0:
            invalid_Ry += 1
    print("Invalid Left X-vals:")
    print(invalid_Lx)
    print("Invalid Left Y-vals:")
    print(invalid_Ly)
    print("Invalid Right X-vals:")
    print(invalid_Rx)
    print("Invalid Right Y-vals:")
    print(invalid_Ry) 



def demo_pix_match():
    #load images
    ref_file = '000POS000Rekonstruktion030.pcf'
    folder = './test_data/testset0/240312_fruit/'
    imgL,imgR = scr.load_images_1_dir(folder, 'cam1', 'cam2', ext = '.jpg')
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(folder + ref_file)
    #set inspection location
    num_chk = 0
    inspec_loc_left = [int(xy1[num_chk][1]),int(xy1[num_chk][0])]
    inspec_loc_right = [int(xy2[num_chk][1]),int(xy2[num_chk][0])]
    #Get neighbouring values
    neighbors_loc_left = np.asarray([[inspec_loc_left[0]-1,inspec_loc_left[1]-1],[inspec_loc_left[0],inspec_loc_left[1]-1],[inspec_loc_left[0]+1,inspec_loc_left[1]-1],
                                [inspec_loc_left[0]-1,inspec_loc_left[1]],[inspec_loc_left[0],inspec_loc_left[1]],[inspec_loc_left[0]+1,inspec_loc_left[1]],
                                [inspec_loc_left[0]-1,inspec_loc_left[1]+1],[inspec_loc_left[0],inspec_loc_left[1]+1],[inspec_loc_left[0]-1,inspec_loc_left[1]+1]])
    neighbors_loc_right = np.asarray([[inspec_loc_right[0]-1,inspec_loc_right[1]-1],[inspec_loc_right[0],inspec_loc_right[1]-1],[inspec_loc_right[0]+1,inspec_loc_right[1]-1],
                                [inspec_loc_right[0]-1,inspec_loc_right[1]],[inspec_loc_right[0],inspec_loc_right[1]],[inspec_loc_right[0]+1,inspec_loc_right[1]],
                                [inspec_loc_right[0]-1,inspec_loc_right[1]+1],[inspec_loc_right[0],inspec_loc_right[1]+1],[inspec_loc_right[0]-1,inspec_loc_right[1]+1]])

    #create left and right pixel stacks at inspection location
    stackL = imgL[:,inspec_loc_left[0],inspec_loc_left[1]]
    stackR = imgR[:,inspec_loc_right[0],inspec_loc_right[1]]
    stackL = stackL[:,np.newaxis]
    stackR = stackR[:,np.newaxis]
    #display left and right pixel stacks

    min_co = np.min([np.min(stackL), np.min(stackR)])
    max_co = np.max([np.max(stackL), np.max(stackR)])
    f = plt.figure()
    f.add_subplot(1,2,1)
    plt.xticks([])
    plt.imshow(stackL, vmin=min_co, vmax= max_co, cmap = "gray")
    f.add_subplot(1,2,2)
    plt.xticks([])
    plt.imshow(stackR, vmin=min_co, vmax= max_co, cmap = "gray")
    plt.colorbar()
    #get neighbouring values
    neiL = np.zeros((imgL.shape[0],neighbors_loc_left.shape[0]))
    neiR = np.zeros((imgL.shape[0],neighbors_loc_right.shape[0]))
    for i in range(neighbors_loc_left.shape[0]):
        po = neighbors_loc_left[i]
        neiL[:,i] = imgL[:,po[0],po[1]]
    for i in range(neighbors_loc_right.shape[0]):
        po = neighbors_loc_right[i]
        neiR[:,i] = imgR[:,po[0],po[1]]
        
        
    #display neighbouring values
    f = plt.figure()
    f.add_subplot(1,2,1)
    
    plt.imshow(neiL, vmin=min_co, vmax= max_co, cmap = "gray")
    
    f.add_subplot(1,2,2)
    plt.imshow(neiR, vmin=min_co, vmax= max_co, cmap = "gray")
    plt.colorbar()


def find_close_stack_disp():
    #attempts to find a pixel stack that is close in range
    match_mode = True
    #load data
    folder = './test_data/testset0/240312_fruit/'
    imgL,imgR = scr.load_images_1_dir(folder, 'cam1', 'cam2', ext = '.jpg')
    ref_file = '000POS000Rekonstruktion030.pcf'
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(folder + ref_file)

    range_thresh = 51
    range_min = 3
    if match_mode:
        for num_chk in tqdm(range(len(xy1))):
            inspec_loc_left = [int(xy1[num_chk][1]),int(xy1[num_chk][0])]
            inspec_loc_right = [int(xy2[num_chk][1]),int(xy2[num_chk][0])]
            stackL = imgL[:,inspec_loc_left[0],inspec_loc_left[1]]
            stackR = imgR[:,inspec_loc_right[0],inspec_loc_right[1]]
            stackL = stackL[:,np.newaxis]
            stackR = stackR[:,np.newaxis]
            min_co = np.min([np.min(stackL), np.min(stackR)])
            max_co = np.max([np.max(stackL), np.max(stackR)])
            co_range = max_co-min_co            
            if co_range < range_thresh and co_range>range_min:
                res_num = num_chk
                break
        print(res_num)
        inspec_loc_left = [int(xy1[res_num][1]),int(xy1[res_num][0])]
        inspec_loc_right = [int(xy2[res_num][1]),int(xy2[res_num][0])]
        #Get neighbouring values
        neighbors_loc_left = np.asarray([[inspec_loc_left[0]-1,inspec_loc_left[1]-1],[inspec_loc_left[0],inspec_loc_left[1]-1],[inspec_loc_left[0]+1,inspec_loc_left[1]-1],
                                    [inspec_loc_left[0]-1,inspec_loc_left[1]],[inspec_loc_left[0],inspec_loc_left[1]],[inspec_loc_left[0]+1,inspec_loc_left[1]],
                                    [inspec_loc_left[0]-1,inspec_loc_left[1]+1],[inspec_loc_left[0],inspec_loc_left[1]+1],[inspec_loc_left[0]-1,inspec_loc_left[1]+1]])
        neighbors_loc_right = np.asarray([[inspec_loc_right[0]-1,inspec_loc_right[1]-1],[inspec_loc_right[0],inspec_loc_right[1]-1],[inspec_loc_right[0]+1,inspec_loc_right[1]-1],
                                    [inspec_loc_right[0]-1,inspec_loc_right[1]],[inspec_loc_right[0],inspec_loc_right[1]],[inspec_loc_right[0]+1,inspec_loc_right[1]],
                                    [inspec_loc_right[0]-1,inspec_loc_right[1]+1],[inspec_loc_right[0],inspec_loc_right[1]+1],[inspec_loc_right[0]-1,inspec_loc_right[1]+1]])

        #create left and right pixel stacks at inspection location
        stackL = imgL[:,inspec_loc_left[0],inspec_loc_left[1]]
        stackR = imgR[:,inspec_loc_right[0],inspec_loc_right[1]]
        stackL = stackL[:,np.newaxis]
        stackR = stackR[:,np.newaxis]
        #display left and right pixel stacks

        min_co = np.min([np.min(stackL), np.min(stackR)])
        max_co = np.max([np.max(stackL), np.max(stackR)])
        f = plt.figure()
        f.add_subplot(1,2,1)
        plt.xticks([])
        plt.imshow(stackL, vmin=min_co, vmax= max_co, cmap = "gray")
        f.add_subplot(1,2,2)
        plt.xticks([])
        plt.imshow(stackR, vmin=min_co, vmax= max_co, cmap = "gray")
        plt.colorbar()
        #get neighbouring values
        neiL = np.zeros((imgL.shape[0],neighbors_loc_left.shape[0]))
        neiR = np.zeros((imgL.shape[0],neighbors_loc_right.shape[0]))
        for i in range(neighbors_loc_left.shape[0]):
            po = neighbors_loc_left[i]
            neiL[:,i] = imgL[:,po[0],po[1]]
        for i in range(neighbors_loc_right.shape[0]):
            po = neighbors_loc_right[i]
            neiR[:,i] = imgR[:,po[0],po[1]]
            
            
        #display neighbouring values
        f = plt.figure()
        f.add_subplot(1,2,1)
        
        plt.imshow(neiL, vmin=min_co, vmax= max_co, cmap = "gray")
        
        f.add_subplot(1,2,2)
        plt.imshow(neiR, vmin=min_co, vmax= max_co, cmap = "gray")
        plt.colorbar()    
    else:
        
        for a in tqdm(range(imgL[0].shape[0]-1)):
            for b in range(imgL[0].shape[1]-1):
                inspec_loc = [a,b]
            
                #create left and right pixel stacks at inspection location
                stackL = imgL[:,inspec_loc[0],inspec_loc[1]]
                stackR = imgR[:,inspec_loc[0],inspec_loc[1]]
                stackL = stackL[:,np.newaxis]
                stackR = stackR[:,np.newaxis]
                min_co = np.min([np.min(stackL), np.min(stackR)])
                max_co = np.max([np.max(stackL), np.max(stackR)])
                co_range = max_co-min_co            
                if co_range < range_thresh and co_range>range_min:
                    res = inspec_loc
                    break
        print(res)
        stackL = imgL[:,res[0],res[1]]
        stackR = imgR[:,res[0],res[1]]
        stackL = stackL[:,np.newaxis]
        stackR = stackR[:,np.newaxis]
        min_co = np.min([np.min(stackL), np.min(stackR)])
        max_co = np.max([np.max(stackL), np.max(stackR)])
        f = plt.figure()
        f.add_subplot(1,2,1)
        plt.xticks([])
        plt.imshow(stackL, vmin=min_co, vmax=max_co, cmap = "gray")
        f.add_subplot(1,2,2)
        plt.xticks([])
        plt.imshow(stackR, vmin=min_co, vmax=max_co, cmap = "gray")
        plt.colorbar()
        #get neighbouring values
        neighbors_loc = np.asarray([[res[0]-1,res[1]-1],[res[0],res[1]-1],[res[0]+1,res[1]-1],
                                    [res[0]-1,res[1]],[res[0],res[1]],[res[0]+1,res[1]],
                                    [res[0]-1,res[1]+1],[res[0],res[1]+1],[res[0]-1,res[1]+1]])
        neiL = np.zeros((imgL.shape[0],neighbors_loc.shape[0]))
        neiR = np.zeros((imgL.shape[0],neighbors_loc.shape[0]))
        for i in range(neighbors_loc.shape[0]):
            po = neighbors_loc[i]
            neiL[:,i] = imgL[:,po[0],po[1]]
            neiR[:,i] = imgR[:,po[0],po[1]]
            
            
        #display neighbouring values
        f = plt.figure()
        f.add_subplot(1,2,1)
        
        plt.imshow(neiL, vmin=min_co, vmax=max_co, cmap = "gray")
        
        f.add_subplot(1,2,2)
        plt.imshow(neiR, vmin=min_co, vmax=max_co, cmap = "gray")
        plt.colorbar()
        

def demo_pix_stack():
    #load images
    folder = './test_data/testset0/240312_fruit/'
    imgL,imgR = scr.load_images_1_dir(folder, 'cam1', 'cam2', ext = '.jpg')
    #set inspection location
    inspec_loc = [780,500]
    #Get neighbouring values
    neighbors_loc = np.asarray([[inspec_loc[0]-1,inspec_loc[1]-1],[inspec_loc[0],inspec_loc[1]-1],[inspec_loc[0]+1,inspec_loc[1]-1],
                                [inspec_loc[0]-1,inspec_loc[1]],[inspec_loc[0],inspec_loc[1]],[inspec_loc[0]+1,inspec_loc[1]],
                                [inspec_loc[0]-1,inspec_loc[1]+1],[inspec_loc[0],inspec_loc[1]+1],[inspec_loc[0]-1,inspec_loc[1]+1]])

    #create left and right pixel stacks at inspection location
    stackL = imgL[:,inspec_loc[0],inspec_loc[1]]
    stackR = imgR[:,inspec_loc[0],inspec_loc[1]]
    stackL = stackL[:,np.newaxis]
    stackR = stackR[:,np.newaxis]
    #display left and right pixel stacks\
    min_co = np.min([np.min(stackL), np.min(stackR)])
    max_co = np.max([np.max(stackL), np.max(stackR)])
    f = plt.figure()
    f.add_subplot(1,2,1)
    plt.xticks([])
    plt.imshow(stackL, vmin=min_co, vmax=max_co, cmap = "gray")
    f.add_subplot(1,2,2)
    plt.xticks([])
    plt.imshow(stackR, vmin=min_co, vmax=max_co, cmap = "gray")
    plt.colorbar()
    #get neighbouring values
    neiL = np.zeros((imgL.shape[0],neighbors_loc.shape[0]))
    neiR = np.zeros((imgL.shape[0],neighbors_loc.shape[0]))
    for i in range(neighbors_loc.shape[0]):
        po = neighbors_loc[i]
        neiL[:,i] = imgL[:,po[0],po[1]]
        neiR[:,i] = imgR[:,po[0],po[1]]
        
        
    #display neighbouring values
    f = plt.figure()
    f.add_subplot(1,2,1)
    
    plt.imshow(neiL, vmin=min_co, vmax=max_co, cmap = "gray")
    
    f.add_subplot(1,2,2)
    plt.imshow(neiR, vmin=min_co, vmax=max_co, cmap = "gray")
    plt.colorbar()

def rect_demo():
    #load images
    image_folder = './test_data/testset0/240312_boat/'
    imgLInd = "cam1"
    imgRInd = "cam2"
    imgL,imgR = scr.load_images_1_dir(image_folder, imgLInd, imgRInd)
    #load f matrix
    f_mat = np.loadtxt("./test_data/testset0/matrices/f.txt", skiprows=2, delimiter = " ")
    rectL, rectR, H1, H2 = scr.rectify_pair(imgL[0],imgR[0], f_mat)
 
    #display in stereo for comparison
    scr.display_4_comp(imgL[0], imgR[0], rectL, rectR)






def triangulate(pt1,pt2,R,t,kL,kR):
    #Create calc matrices 
    Al = np.c_[kL, np.asarray([[0],[0],[0]])]
    

    RT = np.c_[R, t]

    Ar = kR @ RT

    sol0 = pt1[1] * Al[2,:] - Al[1,:]
    sol1 = -pt1[0] * Al[2,:] + Al[0,:]
    sol2 = pt2[1] * Ar[2,:] - Ar[1,:]
    sol3 = -pt2[0] * Ar[2,:] + Ar[0,:]
    
    solMat = np.stack((sol0,sol1,sol2,sol3))
    #Apply SVD to solution matrix to find triangulation
    U,s,vh = np.linalg.svd(solMat,full_matrices = True)

    Q = vh[3,:]

    Q /= Q[3]
    return Q[0:3]
def mag(ve):
    s = 0
    for i in ve:
        s+= i * i
    return np.sqrt(s)


def calib_distort_test():
    #Load calibration images
    cal_folder = "./test_data/cam_cal/calibration_grids/4mm/"
    left_mark = "cam1"
    right_mark = "cam2"
    ext = ".jpg"
    world_scaling = 0.004
    rows =10
    columns = 9
    #Run calibration
    mtx1, mtx2, dist_1, dist_2, R, T, E, F = scr.calibrate_cameras(cal_folder, left_mark, right_mark, ext, rows, columns, world_scaling)
    
    #load images to check size
    data_folder = './test_data/testset0/240312_fruit/'
    imagesL,imagesR = scr.load_images_1_dir(data_folder, left_mark, right_mark)
    img_dim = imagesL[0]
    ho,wo = img_dim.shape
    ref_file = '000POS000Rekonstruktion030.pcf'
    #Load pcf of reference reconstruction
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(data_folder + ref_file)
    
    #Run undistort function for new camera matrices
    new_mtxL, roi = cv2.getOptimalNewCameraMatrix(mtx1, dist_1, (wo,ho), 1, (wo,ho))
    new_mtxR, roi = cv2.getOptimalNewCameraMatrix(mtx2, dist_2, (wo,ho), 1, (wo,ho))
    #Triangulate known correlated points from pcf using new camera matrices
    inspect_ind =0
    print(mtx1)
    print(new_mtxL)
    print(mtx2)
    print(new_mtxR)
    p1 = xy1[inspect_ind]
    p2 = xy2[inspect_ind]
    res1 = triangulate(p1,p2,R,T, mtx1, mtx2)
    res2 = triangulate(p1,p2,R,T, new_mtxL, new_mtxR)
    res3 = geom_arr[inspect_ind]
    print("Current Triangulation:")
    print(res1)
    print("Test Triangulation:")
    print(res2)
    print('#################')
    print("Reference Triangulation")
    print(res3)
    print('#################')
    print("Error Current:")
    print(res1/res3)
    print("#################")
    print("Error Test:")
    print(res2/res3)
    print("#################")
    full_check = True

    diff_chk1 = []
    diff_chk2 = []
    res_tri1 = []
    res_tri2 = []
    avg_diff1 = np.asarray([0.0,0.0,0.0])
    avg_diff2 = np.asarray([0.0,0.0,0.0])
    if full_check:
        for i in tqdm(range(len(xy1))):
            p1 = xy1[i]
            p2 = xy2[i]
            res1 = triangulate(p1,p2,R,T, mtx1, mtx2)
            
            res_tri1.append(res1)
            
            res3 = geom_arr[i]
            diff_chk1.append(res1/res3)
            
            avg_diff1+=res1/res3
            
        for i in tqdm(range(len(xy1))):
            p1 = xy1[i]
            p2 = xy2[i]
            res2 = triangulate(p1,p2,R,T, new_mtxL, new_mtxR)
            res_tri2.append(res2)
            res3 = geom_arr[i]
            diff_chk2.append(res2/res3)
            avg_diff2+=res2/res3
        avg_diff1/=len(xy1)
        print('\n')
        print('Average Error Current:')
        print(avg_diff1)
        avg_diff2/=len(xy1)
        print('Average Error Test:')
        print(avg_diff2)
    res_tri1 = np.asarray(res_tri1)
    res_tri2 = np.asarray(res_tri2)
    
    scr.create_ply(res_tri1)
    scr.create_ply(res_tri2, file_name = "testing2.ply")


def visual_tri_diff():
    data_folder = './test_data/testset0/240312_fruit/'
    images = scr.load_all_imgs_1_dir(data_folder, ext = '.jpg')
    #load pcf of known good data
    ref_file = '000POS000Rekonstruktion030.pcf'
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(data_folder + ref_file)
    mat_folder = './test_data/testset0/matrices/'
    kL, kR, R, t = scr.load_mats(mat_folder)
    

    diffX = []
    diffY = []
    diffZ = []
    diff_mag = []
    for i in tqdm(range(len(xy1))):
        p1 = xy1[i]
        
        p2 = xy2[i]
        res1 = triangulate(p1,p2,R,t, kL, kR)
        res2 = geom_arr[i]
        diff = res2 - res1
        diffX.append(diff[0])
        diffY.append(diff[1])
        diffZ.append(diff[2])
        diff_mag.append(mag(diff))
    res_mapX = np.zeros((images[0].shape[0],images[0].shape[1]), dtype = 'uint8')
    res_mapY = np.zeros((images[0].shape[0],images[0].shape[1]), dtype = 'uint8')
    res_mapZ = np.zeros((images[0].shape[0],images[0].shape[1]), dtype = 'uint8')
    res_map_mag = np.zeros((images[0].shape[0],images[0].shape[1]), dtype = 'uint8')
    for i in range(len(xy1)):
        res_mapX[int(xy1[i][1]),int(xy1[i][0])] = diffX[i]*60
        res_mapY[int(xy1[i][1]),int(xy1[i][0])] = diffY[i]*60
        res_mapZ[int(xy1[i][1]),int(xy1[i][0])] = diffZ[i]*60
        res_map_mag[int(xy1[i][1]),int(xy1[i][0])] = diff_mag[i]*60

    plt.imshow(res_mapX) 
    plt.show()
    plt.imshow(res_mapY) 
    plt.show()
    plt.imshow(res_mapZ) 
    plt.show()
    plt.imshow(res_map_mag) 
    plt.show()

def verif_rect2():
    data_folder = './test_data/testset0/240312_angel/'
    mat_folder = './test_data/testset0/matrices/'
    kL, kR, R, t = scr.load_mats(mat_folder)
    f_file = 'f.txt'
    f_mat = np.loadtxt(mat_folder + f_file, skiprows=2, delimiter = ' ')
    #load images
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgL, imgR = scr.load_images_1_dir(data_folder, imgLInd, imgRInd)
    #calculate rectification homographies
    config = chand.ConfigHandler()
    config.sing_img_folder = data_folder
    config.mat_folder = mat_folder
    config.f_mat_file_mode = 1
    img1,img2, HL, HR = scr.rectify_pair(imgL[0], imgR[0], f_mat)
    xy1,xy2,a1,a2 = ncc.cor_pts(config)
    inspect_ind = 10
    p1 = np.asarray(xy1[inspect_ind])
    p2 = np.asarray(xy2[inspect_ind])
    a1_z = a1[inspect_ind]
    a2_z = a2[inspect_ind]
    print(a1_z)
    print(a2_z)
    rect_p1 = HL @ np.asarray([[p1[0]],[p1[1]], [1]])
    rect_p2 = HR @ np.asarray([[p2[0]],[p2[1]], [1]])
    q = [rect_p1[0,0],rect_p2[0,0], 0, [0,0], rect_p1[1,0]]#subpix is [y,x]
    print('Rectified y_values:')
    print(rect_p1[1,0])
    print(rect_p2[1,0])
    print('Difference: Ideal = 0')
    print(rect_p2[1,0] - rect_p1[1,0] - q[3][0])
    print('####################')
    
    #apply reverse rectification to rectified values
    hL_inv = np.linalg.inv(HL)
    hR_inv = np.linalg.inv(HR)
    
     
    sL = HL[2,0]*q[0] + HL[2,1] * (q[4]) + HL[2,2]
    pL = hL_inv @ np.asarray([[q[0]],[q[4]],[sL]])
    sR = HR[2,0]*(q[1] + q[3][1]) + HR[2,1] * (q[4]+q[3][0]) + HR[2,2]
    pR = hR_inv @ np.asarray([[q[1]+ q[3][1]],[q[4]+q[3][0]],[sR]])
    
    pL = np.asarray([pL[0,0],pL[1,0]])
    pR = np.asarray([pR[0,0],pR[1,0]])
    print('Original Left Camera Point:')
    print(p1)
    print('Rectify Cycled Left Camera Point:')
    print(pL)
    print('############################')
    print('Original Right Camera Point:')
    print(p2)
    print('Rectify Cycled Right Camera Point:')
    print(pR)
    
    print('#############################')
    tot_diff = 0
    min_diff = 2000
    max_diff = 0
    pL_diff = 0
    pR_diff = 0
    for i in range(len(xy1)):
        pt1 = xy1[i]
        pt2 = xy2[i]
        rect_pt1 = HL @ np.asarray([[pt1[0]],[pt1[1]], [1]])
        rect_pt2 = HR @ np.asarray([[pt2[0]],[pt2[1]], [1]])
        diff_chk = np.abs(rect_pt2[1,0] - rect_pt1[1,0])
        tot_diff += diff_chk
        if(diff_chk < min_diff):
            min_diff = diff_chk
        if(diff_chk > max_diff):
            max_diff = diff_chk
        q = [rect_p1[0,0],rect_p2[0,0], 0, [0,0], rect_p1[1,0]]
        sL = HL[2,0]*q[0] + HL[2,1] * (q[4]) + HL[2,2]
        pL = hL_inv @ np.asarray([[q[0]],[q[4]],[sL]])
        sR = HR[2,0]*(q[1] + q[3][1]) + HR[2,1] * (q[4]+q[3][0]) + HR[2,2]
        pR = hR_inv @ np.asarray([[q[1]+ q[3][1]],[q[4]+q[3][0]],[sR]])
        pL = [pL[0,0],pL[1,0]]
        pR = [pR[0,0],pR[1,0]]
        pL_diff += np.abs(pL-p1)[0]
        pR_diff += np.abs(pR-p2)[0]
        
            
    print('######################################')
    print('Average rectified y-value difference:')
    print(tot_diff/len(xy1))
    print("Minimum rectified y-valure difference:")
    print(min_diff)
    print("Maximum rectified y-valure difference:")
    print(max_diff)
    print("Average rectify cycle differences in x values for pL:")
    print(pL_diff/len(xy1))
    print("Average rectify cycle differences in x values for pR:")
    print(pR_diff/len(xy1))
    


def verif_rect():
    data_folder = './test_data/testset0/240312_boat/'
    #data_folder = './test_data/testset0/240411_hand0/'
    #load pcf of known good data
    ref_file = '000POS000Rekonstruktion030.pcf'
    #ref_file = 'pws/000POS000Rekonstruktion030.pcf'
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(data_folder + ref_file)
    #load camera matrices
    mat_folder = './test_data/testset0/matrices/'
    kL, kR, R, t = scr.load_mats(mat_folder)
    f_file = 'f.txt'
    f_mat = np.loadtxt(mat_folder + f_file, skiprows=2, delimiter = ' ')
    #load images
    imgLInd = 'cam1'
    imgRInd = 'cam2'
    imgL, imgR = scr.load_images_1_dir(data_folder, imgLInd, imgRInd)
    #calculate rectification homographies
    img1,img2, HL, HR = scr.rectify_pair(imgL[0], imgR[0], f_mat)
    #apply rectification homographies to known good points
    inspect_ind = 7
    p1 = xy1[inspect_ind]
    p2 = xy2[inspect_ind]
    
    rect_p1 = HL @ np.asarray([[p1[0]],[p1[1]], [1]])
    rect_p2 = HR @ np.asarray([[p2[0]],[p2[1]], [1]])
    q = [rect_p1[0,0],rect_p2[0,0], 0, [0,0], rect_p1[1,0]]#subpix is [y,x]
    print('Rectified y_values:')
    print(rect_p1[1,0])
    print(rect_p2[1,0])
    print('Difference: Ideal = 0')
    print(rect_p2[1,0] - rect_p1[1,0] - q[3][0])
    print('####################')
    
    #apply reverse rectification to rectified values
    hL_inv = np.linalg.inv(HL)
    hR_inv = np.linalg.inv(HR)
    
     
    sL = HL[2,0]*q[0] + HL[2,1] * (q[4]) + HL[2,2]
    pL = hL_inv @ np.asarray([[q[0]],[q[4]],[sL]])
    sR = HR[2,0]*(q[1] + q[3][1]) + HR[2,1] * (q[4]+q[3][0]) + HR[2,2]
    pR = hR_inv @ np.asarray([[q[1]+ q[3][1]],[q[4]+q[3][0]],[sR]])
    
    pL = [pL[0,0],pL[1,0]]
    pR = [pR[0,0],pR[1,0]]
    print('Original Left Camera Point:')
    print(p1)
    print('Rectify Cycled Left Camera Point:')
    print(pL)
    print('############################')
    print('Original Right Camera Point:')
    print(p2)
    print('Rectify Cycled Right Camera Point:')
    print(pR)
    
    print('#############################')
    #confirm matching points
    print('Point Diffs After Cycle')
    print(pL-p1)
    print(pR-p2)
    print('########################')
    #triangulate known good points
    res1 = scr.triangulate(p1,p2,R,t, kL, kR)
    res2 = scr.triangulate(pL,pR,R,t,kL,kR)
    #compare to reference triangulation position
    print('Reference Triangulation:')
    print(geom_arr[inspect_ind])
    print('Triangulation of reference matched points:')
    print(res1)
    print('Triangulation of matched points after rectification cycle:')
    print(res2)
    
    tot_diff = 0
    min_diff = 2000
    max_diff = 0
    pL_diff = 0
    pR_diff = 0
    for i in range(len(xy1)):
        pt1 = xy1[i]
        pt2 = xy2[i]
        rect_pt1 = HL @ np.asarray([[pt1[0]],[pt1[1]], [1]])
        rect_pt2 = HR @ np.asarray([[pt2[0]],[pt2[1]], [1]])
        diff_chk = np.abs(rect_pt2[1,0] - rect_pt1[1,0])
        tot_diff += diff_chk
        if(diff_chk < min_diff):
            min_diff = diff_chk
        if(diff_chk > max_diff):
            max_diff = diff_chk
        q = [rect_p1[0,0],rect_p2[0,0], 0, [0,0], rect_p1[1,0]]
        sL = HL[2,0]*q[0] + HL[2,1] * (q[4]) + HL[2,2]
        pL = hL_inv @ np.asarray([[q[0]],[q[4]],[sL]])
        sR = HR[2,0]*(q[1] + q[3][1]) + HR[2,1] * (q[4]+q[3][0]) + HR[2,2]
        pR = hR_inv @ np.asarray([[q[1]+ q[3][1]],[q[4]+q[3][0]],[sR]])
        pL = [pL[0,0],pL[1,0]]
        pR = [pR[0,0],pR[1,0]]
        pL_diff += np.abs(pL-p1)[0]
        pR_diff += np.abs(pR-p2)[0]
        
            
    print('######################################')
    print('Average rectified y-value difference:')
    print(tot_diff/len(xy1))
    print("Minimum rectified y-valure difference:")
    print(min_diff)
    print("Maximum rectified y-valure difference:")
    print(max_diff)
    print("Average rectify cycle differences in x values for pL:")
    print(pL_diff/len(xy1))
    print("Average rectify cycle differences in x values for pR:")
    print(pR_diff/len(xy1))




def pre_demo():#demo of preprocessingimage filters and grayscale conversion
    folder = './test_data/testset0/240411_hand1/'
    folder = './test_data/testset0/240312_boat/'
    matrix_folder = "./test_data/testset0/matrices/"
    #load images
    imgL,imgR = scr.load_images_1_dir(folder, 'cam1', 'cam2', ext = '.jpg')
    imgL_u, imgR_u = scr.load_images_1_dir(folder, 'cam1', 'cam2', ext = '.jpg', colorIm = True)
    #Display unaltered images
    scr.display_stereo(imgL_u[0], imgR_u[0])
    #display grayscale
    scr.display_stereo(imgL[0],imgR[0])
    fund_mat = np.loadtxt(matrix_folder + "f.txt", skiprows=2, delimiter = " ")
    #Display rectified
    rectL,rectR = scr.rectify_lists(imgL,imgR, fund_mat)
    scr.display_stereo(rectL[0],rectR[0])
    #display filtered
    avgL = np.asarray(rectL).mean(axis=0)
    avgR = np.asarray(rectR).mean(axis=0)
    thresh_val = 50
    maskL = scr.mask_avg_list(avgL,rectL, thresh_val)
    maskR = scr.mask_avg_list(avgR,rectR, thresh_val)
    #Display isolation
    scr.display_stereo(maskL[0],maskR[0])
    offL = 80
    offR = 80
    offT = 400
    offB = 60
    maskL = np.asarray(maskL).astype("uint8")
    maskR = np.asarray(maskR).astype("uint8")
    scr.create_stereo_offset_fig_internal(maskL[0],maskR[0],offL, offR, offT, offB)



def demo_sift():
    #load images

    folder1 = './test_data/250221_Cudatest/pos9/'
    imgL,imgR = scr.load_images_1_dir(folder1, 'cam1', 'cam2', ext = '.jpg', colorIm = True)
    print(imgL[0].shape)
    #apply SIFT
    pts1,pts2,col,F = scr.feature_corr(imgL[1],imgR[1])
    #draw found matching points on stereo
    scr.mark_points(imgL[0],imgR[0],pts1,pts2,size = 20,showBox = False)


def disp_cal():
    #load images
    ext = '.jpg'
    folder = './test_data/cam_cal/calibration_grids/4mm/'
    images = scr.load_images_basic(folder, ext)
    plt.imshow(images[0])
    plt.show()
    print(images[0].shape)

    '''
    lwr = np.array([0, 0, 143])
    upr = np.array([179, 61, 252])
    hsv = cv2.cvtColor(images[0], cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(hsv, lwr, upr)
    print(msk.shape)
    plt.imshow(msk, cmap = 'gray')
    plt.show()
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dlt = cv2.dilate(msk, krn, iterations=5)
    res = 255 - cv2.bitwise_and(dlt, msk)
    res = np.uint8(res)
    plt.imshow(res, cmap = 'gray')
    plt.show()
    '''
    
    
    #set constants
    world_scaling = 0.004
    rows =10
    columns = 9
    #preprocess images
    prep_img = []
    for i in range(0,19):
        frame = images[i]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_thr = int(gray.max()*0.8)
        mask1 = np.ones_like(gray)
        mask1[gray < mask_thr] = 0 
        gray = gray*mask1
        
        prep_img.append(gray)
        
        
        
        

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
    width = images[0].shape[1]
    height = images[0].shape[0]
    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
    chkfrm_list = []
    for i in tqdm(range(len(prep_img))):

        gray = prep_img[i]

        #find the checkerboard
        
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), cv2.CALIB_CB_ADAPTIVE_THRESH)
        
        if ret == True:
            
            
            conv_size = (11, 11)
 
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            checkframe = cv2.drawChessboardCorners(images[i], (rows,columns), corners, ret)
            chkfrm_list.append(checkframe)
            objpoints.append(objp)
            imgpoints.append(corners)
    print("Resolving Calibration...")    
    try:    
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    except Exception as e:
        print(e)
        print("Calibration Failure.")
        mtx = None
        dist = None
    print(mtx)
    print(dist)
    print(len(chkfrm_list)) 
    f = plt.figure()
    f.set_figwidth(60)
    f.set_figheight(40)
    plt.imshow(chkfrm_list[2])
    plt.show()
      


    

   




       
def compare_f_mat_search():
    #reference fmat
    data_folder = './test_data/testset0/'
    matloc = data_folder + 'matrices/f.txt'
    ref_mat = np.loadtxt(matloc, skiprows=2, delimiter = ' ')
    print('Ref:')
    print(ref_mat)
    #load image stacks
    imgsL,imgsR = scr.load_images_1_dir(data_folder + "240312_fruit/","cam1","cam2", ".jpg")
    
    #use ncc search
    f_ncc = scr.find_f_mat_ncc(imgsL,imgsR, thresh = 0.6)
    print('NCC:')
    print(f_ncc)
    #use SIFT search with LMEDS
    f_sift = scr.find_f_mat(imgsL[0], imgsR[0])
    print('LMEDS:')
    print(f_sift)

    ref_L, ref_R, H1, H2 = scr.rectify_pair(imgsL[0], imgsR[0], ref_mat)
    scr.display_stereo(imgsL[0],imgsR[0])
    scr.display_stereo(ref_L,ref_R)
    ncc_L, ncc_R, H1, H2 = scr.rectify_pair(imgsL[0], imgsR[0], f_ncc)
    scr.display_stereo(ncc_L,ncc_R)
    sift_L, sift_R, H1, H2 = scr.rectify_pair(imgsL[0], imgsR[0], f_sift)
    scr.display_stereo(sift_L,sift_R)





def test_thr():
    # Create Object 
    root = tk.Tk() 
  
    # Set geometry 
    root.geometry("400x400") 
    global stop_threads
    stop_threads = False
    def run():
        n=200e10
        i = 0
        while i < n:
            print('thread running')
            i+= 1
            print(i)
            global stop_threads
            if stop_threads:
                break
    global t1
    t1 = thr.Thread(target = run)
    def stop():
        global stop_threads
        stop_threads = True
        t1.join()
        print('CANCELLED')
    
    def start():
        
        t1.start()
    # Create Button 
    tk.Button(root,text="Start",command = start).pack() 
    tk.Button(root,text="Cancel",command = stop).pack() 
    # Execute Tkinter 
    root.mainloop()

def conv_pcf_ply():
    pcf_loc = './test_data/testset0/240411_hand1/pws/000POS000Rekonstruktion030.pcf'
    out_file = 'ref_hand1.ply'
    scr.pcf_to_ply(pcf_loc, out_file)


def test_get_RT():
    #Load kL,kR,F
    kL_file = 'Kl.txt'
    kR_file = 'Kr.txt'
    f_file = 'f.txt'
    r_file = 'R.txt'
    t_file = 't.txt'
    folder = './test_data/testset0/matrices/'
    skiprow = 2
    delim = ' '
    kL = np.loadtxt(folder + kL_file, skiprows=skiprow, delimiter = delim)
    kR = np.loadtxt(folder + kR_file, skiprows=skiprow, delimiter = delim)
    R = np.loadtxt(folder + r_file, skiprows=skiprow, delimiter = delim)
    t = np.loadtxt(folder + t_file, skiprows=skiprow, delimiter = delim)
    F = np.loadtxt(folder + f_file, skiprows=skiprow, delimiter = delim)
    inputfile = './test_data/testset0/240312_boat/000POS000Rekonstruktion030.pcf'
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(inputfile)
    #Convert F to E
    E = kR.T @ F @ kL
    #Decompose E to R and t
    R1,R2,t1 = cv2.decomposeEssentialMat(E)
    #print results
    print('Using known F')
    print('Known F:')
    print(F)
    print('Known R:')
    print(R)
    print('Known t:')
    print(t)
    print('Calc R1:')
    print(R1)
    print('Calc R2:')
    print(R2)
    print('Calc t:')
    print(t1)
    a,Rrec,Trec,b = cv2.recoverPose(E,xy1,xy2)
    
    print('Recovered R:')
    print(Rrec)
    print('Recovered T:')
    print(Trec)
    print('Using calculated F')
    #load images
    img_folder = './test_data/testset0/240312_boat/'
    imgL,imgR = scr.load_images_1_dir(img_folder, 'cam1', 'cam2', '.jpg')
    #run find f mat with LMEDS SIFT
    Fcalc = scr.find_f_mat(imgL[0], imgR[0])
    print('Calc F:')
    print(Fcalc)
    #calc E mat
    Ecalc = kR.T @ Fcalc @ kL
    #Decompose E
    R1c,R2c,t1c = cv2.decomposeEssentialMat(Ecalc)
    print('Calc F R1:')
    print(R1c)
    print('Calc F R2:')
    print(R2c)
    print('Calc F t:')
    print(t1c)
    
    a,Rrec1,Trec1,b = cv2.recoverPose(Ecalc,xy1,xy2)
    
    print('Recovered calc F R:')
    print(Rrec1)
    print('Recovered calc F T:')
    print(Trec1)
    
    

def simple_idw(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi

def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])

    return np.hypot(d0, d1)

def linear_rbf(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)

    # Mutual pariwise distances between observations
    internal_dist = distance_matrix(x,y, x,y)

    # Now solve for the weights such that mistfit at the observations is minimized
    weights = np.linalg.solve(internal_dist, z)

    # Multiply the weights for each interpolated point by the distances
    zi =  np.dot(dist.T, weights)
    return zi

def comb_lin_rbf(x, y, z, xi, yi):
    obs = np.vstack((x, y)).T
    interp = np.vstack((xi, yi)).T
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])
    dist = np.hypot(d0, d1)
    
    interp0 = np.vstack((x, y)).T
    d0 = np.subtract.outer(obs[:,0], interp0[:,0])
    d1 = np.subtract.outer(obs[:,1], interp0[:,1]) 
    internal_dist = np.hypot(d0, d1)
    
    weights = np.linalg.solve(internal_dist, z)
    zi =  np.dot(dist.T, weights)
    return zi

def search_interp_field(grid,x,y,x_min, x_max, y_min, y_max, n):
    #This function retrieves the closest value to the point requested 
    #in a grid with the dimensions of xmin to xmax, ymin to ymax and the interpolation number n
    #Calculate based on splitting the ranges into n what index to access
    x_tot = x_max - x_min
    y_tot = y_max - y_min
    
    x_inc = x_tot/n
    y_inc = y_tot/n
    
    x_ind = int(np.round(x/x_inc))
    y_ind = int(np.round(y/y_inc))
    
    print(x_ind)
    print(y_ind)
    return grid[x_ind,y_ind]
    
def scipy_rbf(x, y, z, xi, yi):
    interp = Rbf(x, y, z, function='linear')
    return interp(xi, yi)
def plotSP(x,y,z,grid, show_points = True):
    plt.figure()
    plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()))
    if(show_points):
        plt.scatter(x,y,c=z, edgecolors = 'black')
    plt.colorbar()
    
def test_interp0():
    # Setup: Generate data
    n = 3
    nx, ny = 20, 20
    x, y, z = map(np.random.random, [n, n, n])
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    xi, yi = np.meshgrid(xi, yi)
    xi, yi = xi.flatten(), yi.flatten()

    grid3 = linear_rbf(x,y,z,xi,yi)
    grid3 = grid3.reshape((ny, nx))

    plotSP(x,y,z,grid3, False)
    plt.title('Rbf')

    plt.show()

@numba.jit(nopython=True)
def rep_grid(g,h):
    
    g_len = g.shape[0]
    h_len = h.shape[0]
    
    
    resG = []
    resH = []
    for u in range(h_len):
        resG.append(g)
    for u in range(g_len):
        resH.append(h)
    #Todo 
    #- use loops to transpose resH
    resHT = []
    for a in h:
        resHT_R = []
        for b in range(g_len):
           resHT_R.append(a) 
        resHT.append(resHT_R)
    #- use loops to flatten resG and resH DONE
    resFlatG = []
    for i in resG:
        for j in i:
            resFlatG.append(j)
    resFlatH = []
    for i in resHT:
        for j in i:
            resFlatH.append(j)

    print('::::::::::::::::::')  
    testL = np.array(resFlatG)
    testK = np.array(resFlatH)
    test2L = np.vstack((testL,testK))
    print(test2L)
    print('::::::::::::::::::')  
    return resG,resH
       
def test_grid():
    nx, ny = (3,3)
    
    x = np.linspace(0, 1, nx)
    y = np.linspace(1, 2, ny)
    print(x)
    print('#####')
    print(y)
    print('-------------')
    xv, yv = np.meshgrid(x, y)
    print(xv)
    print('=======')
    print(yv)
    print('----------')
    resX,resY = rep_grid(x,y)
    resT = np.vstack([x]*y.shape[0])
    print(resT)
    print(np.asarray(resX))
    print(resT.flatten())
    print('=======')
    resT = np.vstack([y]*x.shape[0]).T
    print(resT)
    print(np.asarray(resY))
    print('########')
    print(np.asarray(resY).T)
    print('########')
    print(resT.flatten())
    print('======')

def check_gpu():
    print(cu.current_context().device.name)

    

def demo_hist():
    folder = "./test_data/"
    data = "Speckleshist.csv"
    bins = []
    counts = []
    with open(folder+data, newline='') as csvfile:
        num_reader = csv.reader(csvfile, delimiter = " ", quotechar="|")
        counter = 0
        for row in num_reader:
            if(counter != 0):
                s = row[0].split(",")
                bins.append(int(s[0]))
                counts.append(int(s[1]))
            counter+=1
            
    plt.bar(bins,counts)
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.title('Histogram of Speckle Pattern Intensities')
    plt.show()


def demo_rbf(interp_num = 3, randZ = False):

    #Generate data - 8-neighbor + Central point = 9 points known for x,y,z
    #Set x and y locations
    #Set z values as well for consistency, but add option to randomize
    x_val = np.asarray([0,0.5,1,0,0.5,1,0,0.5,1])
    y_val = np.asarray([0,0,0,0.5,0.5,0.5,1,1,1])
    z_val = np.asarray([1,0,1,0.5,1,0.5,0.75,0.5,0])
    

    if randZ:
        z_val = np.random.rand(9)
    n = interp_num*2+3
    xi = np.linspace(x_val.min(), x_val.max(), n)
    yi = np.linspace(y_val.min(), y_val.max(), n)
    
  #  xi, yi = np.meshgrid(xi, yi)
  #  xi, yi = xi.flatten(), yi.flatten()
    
  #Meshgrid+flatten replacement
    g_len = xi.shape[0]
    h_len = xi.shape[0]
    
    
    resG = []
    resH = []
    for u in range(h_len):
        resG.append(xi)
    for u in range(g_len):
        resH.append(yi)

    resHT = []
    for a in yi:
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
    xi = np.array(resFlatG)
    yi = np.array(resFlatH)
    
    #linear rbf 
    obs = np.vstack((x_val, y_val)).T
    interp = np.vstack((xi, yi)).T
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])
    dist = np.hypot(d0, d1)
    interp0 = np.vstack((x_val, y_val)).T
    d0 = np.subtract.outer(obs[:,0], interp0[:,0])
    d1 = np.subtract.outer(obs[:,1], interp0[:,1]) 
    internal_dist = np.hypot(d0, d1)
    weights = np.linalg.solve(internal_dist, z_val)
    grid =  np.dot(dist.T, weights)
    
    
    
    grid = grid.reshape((n, n))
    plotSP(x_val,y_val,z_val,grid, False)


    plt.show()
    '''
    print(z_val)
    
    a = 0.6
    b = 0.5
    #search interp field returns incorrect results when looking for known points used to create the interpolation field
    c = search_interp_field(grid,a,b,x_val.min(), x_val.max(),y_val.min(), y_val.max(), n)
    print(c)
    '''


def demo_lin(interp_num = 3, randZ = False):
    #Generate data - 8-neighbor + Central point = 9 points known for x,y,z
    #Set x and y locations
    #Set z values as well for consistency, but add option to randomize
    x_val = np.asarray([0,0.5,1,0,0.5,1,0,0.5,1])
    y_val = np.asarray([0,0,0,0.5,0.5,0.5,1,1,1])
    z_val = np.asarray([1,0,1,0.5,1,0.5,0.75,0.5,0])
    

    if randZ:
        z_val = np.random.rand(9)
    diag_len = 1.41421356237
    

    n = interp_num*2+3

    grid = 0*np.ones((n,n))
    #dim[NW,N,NE,W,C,E,SW,S,SE]
    #ind[0,1,2,3,4,5,6,7,8]   
    #Intended output: Grid with known values placed in respective locations, and linear interpolation calculations on 8-neighbor lines
    #Unknown values are set to -1. 
    grid[0,0] = z_val[0]
    grid[0,int(n/2)] = z_val[1]
    grid[0,n-1] = z_val[2]     
    grid[int(n/2),0] = z_val[3]
    grid[int(n/2),int(n/2)] = z_val[4]
    grid[int(n/2),n-1] = z_val[5]
    grid[(n-1),0] = z_val[6]
    grid[(n-1),int(n/2)] = z_val[7]
    grid[(n-1),(n-1)] = z_val[8]
    #calculate cardinal
    
    increment = 1/(1+interp_num)
    nw_count = 1
    n_count = 1
    ne_count = 1
    w_count = 1
    e_count = 1
    sw_count = 1
    s_count = 1
    se_count = 1
    for i in range(n):
        for j in range(n):
            if(i>0 and i < n-1 and j > 0 and j < n-1):
                if(i == j and i < int(n/2)):
 
                    m = (grid[int(n/2),int(n/2)] - grid[0,0])/diag_len
                    grid[i,j] = m * increment * nw_count + grid[0,0]
                    nw_count += 1
                elif(i== j and i > int(n/2)):

                    m = (grid[int(n/2),int(n/2)] - grid[(n-1),(n-1)])/diag_len
                    grid[i,j] = grid[int(n/2),int(n/2)]- m * increment * se_count
                    se_count += 1
                elif(j > int(n/2) and i < int(n/2) and i == n-1-j):

                    m = (grid[int(n/2),int(n/2)] - grid[0,n-1])/diag_len
                    grid[i,j] = m * increment * ne_count + grid[0,n-1]
                    ne_count += 1
                elif(j < int(n/2) and i > int(n/2) and j == n-1-i):

                    m = (grid[int(n/2),int(n/2)] - grid[(n-1),0])/diag_len
                    grid[i,j] = grid[int(n/2),int(n/2)]-m * increment * sw_count 
                    sw_count += 1
                elif(i == int(n/2) and j < int(n/2)):

                    m = grid[int(n/2),int(n/2)] - grid[int(n/2),0]
                    grid[i,j] = m * increment *w_count + grid[int(n/2),0]
                    w_count += 1
                elif(i == int(n/2) and j > int(n/2)):

                    m = grid[int(n/2),int(n/2)] - grid[int(n/2),n-1]
                    grid[i,j] =grid[int(n/2),int(n/2)]- m * increment  * e_count  
                    e_count += 1
                elif(i > int(n/2) and j == int(n/2)):

                    m = grid[int(n/2),int(n/2)] - grid[(n-1),int(n/2)]
                    grid[i,j] =grid[int(n/2),int(n/2)]- m * increment * s_count
                    s_count += 1
                elif(i < int(n/2) and j == int(n/2)):

                    m = grid[int(n/2),int(n/2)] - grid[0,int(n/2)]
                    grid[i,j] = m * increment * n_count + grid[0,int(n/2)]
                    n_count += 1
    plotSP(x_val,y_val,z_val,grid, False)
    

      
@numba.jit(nopython=True)   
def test_interp_stack():
    #set test 'image stack'
    x_val = np.asarray([0,0.5,1,0,0.5,1,0,0.5,1])
    y_val = np.asarray([0,0,0,0.5,0.5,0.5,1,1,1])
    z_val_list = [[1,0,0.5],[0,0.5,1],[1,0,0.5],[0.5,0,1],[0.5,0,1],[0.5,0,1],[0,0.5,1],[1,0,0.5],[0,0.5,1]]
    #set test pixel stack
    Gi = np.asarray([0.8,0.2,0.6])
    
    #begin test function
    max_cor = 0
    max_mod = [0.0,0.0] #default to no change
    agi = np.sum(Gi)/Gi.shape[0]
    val_i = np.sum((Gi-agi)**2)
    z_val = np.empty((len(z_val_list),len(z_val_list[0])))
    for a in range(len(z_val_list)):
        for b in range(len(z_val_list[0])):
            z_val[a][b] = z_val_list[a][b]
    
    

    n = 9
    xi = np.linspace(x_val.min(), x_val.max(), n)
    yi = np.linspace(y_val.min(), y_val.max(), n)
    g_len = xi.shape[0]
    h_len = xi.shape[0]
    
    
    resG = []
    resH = []
    for u in range(h_len):
        resG.append(xi)
    for u in range(g_len):
        resH.append(yi)

    resHT = []
    for a in yi:
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
    xi = np.array(resFlatG)
    yi = np.array(resFlatH)
    interp_fields_list = []
    for a in range(z_val.shape[1]):
        #linear rbf 
        obs = np.vstack((x_val, y_val)).T
        interp = np.vstack((xi, yi)).T

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
        weights = np.linalg.solve(internal_dist, z_val[:,a])
        grid =  np.dot(dist.T, weights)
        grid = grid.reshape((n, n))
        interp_fields_list.append(grid)
    interp_fields = np.empty((len(interp_fields_list),len(interp_fields_list[0]),len(interp_fields_list[0][0])))
    for a in range(len(interp_fields_list)):
        for b in range(len(interp_fields_list[0])):
            for c in range(len(interp_fields_list[0][0])):
                interp_fields[a][b][c] = interp_fields_list[a][b][c]
    
    
    dist_inc = 1/n
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
                        max_mod = [j*dist_inc, i*dist_inc]
    print(max_mod)

