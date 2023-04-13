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
def access_slice_data(entry,imgL, imgR):
    xyL = entry[0]
    xyR = entry[1]
    valL = []
    valR = []
    float_epsilon = 1e-6
    if(xyL[0] - int(xyL[0]) < float_epsilon  and xyL[1] - int(xyL[1]) < float_epsilon 
       and xyR[0] - int(xyR[0]) < float_epsilon and xyR[1] - int(xyR[1]) < float_epsilon):
        valL = imgL[xyL[0],[xyL[1]],:]
        valR = imgL[xyL[0],[xyL[1]],:]
    else:
        #interpolate to get subpixel values
        for i in range(len(imgL)): 
            #coordinate modifiers from center
            cent_x = int(xyR[0])
            cent_y = int(xyR[1])
            x = [-1,-1,-1,0,0,0,1,1,1]
            y = [-1,0,1,-1,0,1,-1,0,1]
            X_g = np.linspace(-1, 1)
            Y_g = np.linspace(-1, 1)
            X_g, Y_g = np.meshgrid(X_g, Y_g)  # 2D grid for interpolation
            #TODO intensity values at coordinates
            zL = []
            interpL = LinearNDInterpolator(list(zip(x, y)), zL)
            zR = []
            interpR = LinearNDInterpolator(list(zip(x, y)), zR)
            
    return valL,valR
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
        neg_entry[6] = scram_pair[6]
        train_neg.append(neg_entry)
    return train_neg
def split_pairing_data(xyL,xyR,geom_arr,col_arr,correl, imgL, imgR):
    train_pos = [] #0
    train_scram = []#1
    verif = [] #2
    prev_code = -1
    counter_pos = 0
    for i in range(len(xyL)):
        entry = [xyL[i], xyR[i], geom_arr[i], col_arr[i], correl[i]]
        valL, valR = access_slice_data(entry, imgL, imgR)
        entry.append(valL)
        entry.append(valR)
        if(prev_code == -1 or prev_code == 2):
            train_pos.append(entry)
            prev_code = 0
            counter_pos += 1
        elif(prev_code == 0):
            train_scram.append(entry)
            prev_code = 1
        elif(prev_code == 1 and counter_pos < 2):
            train_pos.append(entry)
            prev_code = 0
        elif(prev_code == 1 and counter_pos == 2):
            verif.append(entry)
            prev_code = 2
            counter_pos = 0
    train_neg = scramble_neg_data(train_scram, train_pos, verif)
    return train_pos, train_neg, verif

def script_test():
    folder_statue = "./test_data/statue/"
   #TODO - check if all subpixel location values occur on right-side images
script_test()