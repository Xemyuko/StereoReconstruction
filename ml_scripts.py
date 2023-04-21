# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 20:54:59 2023

@author: myuey
"""
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import scripts as scr
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
import random
float_chk = 1e-9
def generate_neighbors(yC, xC, yLim, xLim):
    x = [-1,-1,-1,0,0,1,1,1]
    y = [-1,0,1,-1,1,-1,0,1]
    res = []
    for i,j in zip(y,x):
        pN = np.asarray([int(yC)+i,int(xC)+j])
        if(pN[0] >= 0 and pN[0] < yLim and pN[1] >= 0 and pN[1]< xLim):
            res.append(pN)
    return np.asarray(res, dtype = 'uint64')
def access_data(img_stack, yC, xC, yLim, xLim):
    val = []
    if(xC - int(xC) <= float_chk and yC - int(yC) <= float_chk):
        val = img_stack[:,int(yC), int(xC)]
        
    else:
        #interpolate to get subpixel values
        for i in range(len(img_stack)):
            cent_x = int(xC)
            cent_y = int(yC)
            x = [-1,-1,-1,0,0,0,1,1,1]
            y = [-1,0,1,-1,0,1,-1,0,1]
            X_g = np.linspace(-1, 1)
            Y_g = np.linspace(-1, 1)
            Y_g, X_g = np.meshgrid(Y_g, X_g)  # 2D grid for interpolation
            z = []
            y_i = []
            x_i = []
            for m,n in zip(x,y):
                xN = cent_x+m
                yN = cent_y+n
                y_i.append(yN)
                x_i.append(xN)
                if(yN >= 0 and yN < yLim and xN >= 0 and xN < xLim):
                    z.append(img_stack[i,yN,xN])
                elif(cent_y >= 0 and cent_y < yLim and cent_x >= 0 and cent_x < xLim):
                    z.append(img_stack[i,cent_y, cent_x])
                else:
                    z.append(0.0)
            interp = LinearNDInterpolator(list(zip(y_i, x_i)), z)
            val.append(interp(yC, xC))
    return np.asarray(val, dtype = 'float32')

def scramble_neg_data(train_scram, train_pos, verif):
    train_neg = []
    total_len = len(train_scram) + len(train_pos) + len(verif)
    for i in range(len(train_scram)):
        rand_range = list(set((range(0,total_len, 1))) - set([i]))
        rand_val = random.choice(rand_range)
        scram_pair = []
        if(rand_val < len(train_scram)):
            scram_pair = train_scram[rand_val]
        elif(rand_val < len(train_scram) + len(train_pos)):
            scram_pair = train_pos[rand_val - len(train_scram)]
        else:
            scram_pair = verif[rand_val - len(train_scram) - len(train_pos)]
        neg_entry = train_scram[i]
        neg_entry[1] = scram_pair[1]
        train_neg.append(neg_entry)
    return np.asarray(train_neg, dtype = 'object')

def split_pairing_data(xyL,xyR,imgL, imgR, yLim, xLim):
    train_pos = [] #0
    train_scram = []#1
    verif = [] #2
    prev_code = -1
    counter_pos = 0
    for i in tqdm(range(len(xyL))):
        yCL = xyL[i][1]
        xCL = xyL[i][0]
        yCR = xyR[i][1]
        xCR = xyR[i][0]
        neighborsL = generate_neighbors(yCL, xCL, yLim, xLim)
        neighborsR = generate_neighbors(yCR, xCR, yLim, xLim)
        entry_data_c_L = access_data(imgL, yCL, xCL, yLim, xLim)
        entry_data_c_R = access_data(imgR, yCR, xCR, yLim, xLim)
        entry_data_n_L = []
        entry_data_n_R = []
        for a,b in zip(neighborsL, neighborsR):
            entry_data_n_L.append(access_data(imgL, a[0], a[1], yLim, xLim))
            entry_data_n_R.append(access_data(imgR, b[0], b[1], yLim, xLim))
        ed_n_L = np.asarray(entry_data_n_L)
        ed_n_R = np.asarray(entry_data_n_R)
        entry = [np.asarray(xyL[i], dtype = 'float32'), 
                            np.asarray(xyR[i], dtype = 'float32'), 
                            neighborsL, neighborsR, entry_data_c_L, entry_data_c_R, 
                            ed_n_L, ed_n_R]
        
        if(prev_code == -1 or prev_code == 2):
            prev_code = 0
            entry.append(prev_code)
            entry = np.asarray(entry,dtype = 'object')
            train_pos.append(entry)
            counter_pos += 1
        elif(prev_code == 0 and counter_pos >= 3):
            prev_code = 1
            entry.append(prev_code)
            entry = np.asarray(entry,dtype = 'object')
            train_scram.append(entry)
            
        elif(prev_code == 0 and counter_pos < 3):
            prev_code = 0
            entry.append(prev_code)
            entry = np.asarray(entry,dtype = 'object')
            train_pos.append(entry)
            counter_pos+=1
            
        elif(prev_code == 1 and counter_pos < 3):
            prev_code = 0
            entry.append(prev_code)
            entry = np.asarray(entry,dtype = 'object')
            train_pos.append(entry)
            counter_pos += 1
            
        elif(prev_code == 1 and counter_pos >= 3):
            prev_code = 2
            entry.append(prev_code)
            entry = np.asarray(entry,dtype = 'object')
            verif.append(entry)
            counter_pos = 0
    train_neg = scramble_neg_data(train_scram, train_pos, verif)
    verif = np.asarray(verif, dtype = 'object')
    train_pos = np.asarray(train_pos, dtype = 'object')
    return train_pos, train_neg, verif
def count_subpixel(xyList):
    counter = 0
    for i in xyList:
        if(i[0] - int(i[0] > float_chk or i[1] - int(i[1]) > float_chk)):
            counter+=1
    return counter
def build_dataset(pcf_file, imgL, imgR, yLim, xLim, train_pos_name = "trainPosC", inc_num = 100,
                  train_neg_name = "trainNegC", verif_name = "verificationC"):
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(pcf_file)
    xy1 = xy1[::inc_num]
    xy2 = xy2[::inc_num]
    train_pos, train_neg, verif = split_pairing_data(xy1, xy2, imgL, imgR, yLim, xLim)
    np.save(train_pos_name,train_pos)
    np.save(train_neg_name,train_neg)
    np.save(verif_name, verif)
def load_dataset(train_pos_name = "trainPosC.npy",
                 train_neg_name = "trainNegC.npy", verif_name = "verificationC.npy"):
    train_pos = np.load(train_pos_name, allow_pickle = True)
    train_neg = np.load(train_neg_name, allow_pickle = True)
    verif = np.load(verif_name, allow_pickle = True)
    return train_pos, train_neg, verif
def script_test():
    folder_statue = "./test_data/statue/"
    left_folder = "camera_L/"
    right_folder = "camera_R/"
    pcf_file = folder_statue + "Rekonstruktion30.pcf"
    imgL,imgR = scr.load_images(folderL = folder_statue+left_folder, folderR = folder_statue+right_folder)
    imshape = imgL[0].shape
    xLim = imshape[1]
    yLim = imshape[0]
    #build_dataset(pcf_file, imgL, imgR,yLim,xLim)
    a,b,c = load_dataset()
    d = np.concatenate((a,b))
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)
    
script_test()