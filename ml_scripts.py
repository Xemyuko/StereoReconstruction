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

def access_data(img_stack, xC, yC):
    val = []
    if(xC - int(xC) <= float_chk and yC - int(yC) <= float_chk):
        val = img_stack[xC, yC, :]
        
    else:
        #interpolate to get subpixel values
        for i in range(len(img_stack)):
            cent_x = int(xC)
            cent_y = int(yC)
            x = [-1,-1,-1,0,0,0,1,1,1]
            y = [-1,0,1,-1,0,1,-1,0,1]
            X_g = np.linspace(-1, 1)
            Y_g = np.linspace(-1, 1)
            X_g, Y_g = np.meshgrid(X_g, Y_g)  # 2D grid for interpolation
            z = []
            for i,j in zip(x,y):
                xN = cent_x+i
                yN = cent_y+j
                z.append(access_data(img_stack), xN, yN)
            interp = LinearNDInterpolator(list(zip(x, y)), z)
            val.append(interp(xC, yC))
    return np.asarray(val)

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
def generate_neighbors(xC, yC):
    x = [-1,-1,-1,0,0,1,1,1]
    y = [-1,0,1,-1,1,-1,0,1]
    res = []
    for i,j in zip(x,y):
        pN = (xC+i,yC+j)
        res.append(pN)
    return res
def split_pairing_data(xyL,xyR,imgL, imgR):
    train_pos = [] #0
    train_scram = []#1
    verif = [] #2
    prev_code = -1
    counter_pos = 0
    for i in range(len(xyL)):
        xCL = xyL[i][0]
        yCL = xyL[i][1]
        xCR = xyR[i][0]
        yCR = xyR[i][1]
        neighborsL = generate_neighbors(xCL, yCL)
        neighborsR = generate_neighbors(xCR, yCR)
        entry_data_c_L = access_data(imgL, xCL, yCL)
        entry_data_c_R = access_data(imgR, xCR, yCR)
        entry_data_n_L = []
        entry_data_n_R = []
        for a,b in zip(neighborsL, neighborsR):
            entry_data_n_L.append(access_data(imgL, a[0], a[1]))
            entry_data_n_R.append(access_data(imgR, b[0], b[1]))
        ed_n_L = np.asarray(entry_data_n_L)
        ed_n_R = np.asarray(entry_data_n_R)
        entry = [xyL[i], xyR[i], neighborsL, neighborsR, entry_data_c_L, entry_data_c_R, ed_n_L, ed_n_R]
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
    verif = np.asarray(verif)
    train_pos = np.asarray(train_pos)
    return train_pos, train_neg, verif
def count_subpixel(xyList):
    counter = 0
    for i in xyList:
        if(i[0] - int(i[0] > float_chk or i[1] - int(i[1]) > float_chk)):
            counter+=1
    return counter
def build_dataset(pcf_file, imgL, imgR, train_pos_name = "trainPos.txt",
                  train_neg_name = "trainNeg.txt", verif_name = "verification.txt"):
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(pcf_file)
    train_pos, train_neg, verif = split_pairing_data(xy1, xy2, imgL, imgR)
    np.savetxt(train_pos_name,train_pos)
    np.savetxt(train_neg_name,train_neg)
    np.savetxt(verif_name, verif)
def load_dataset(train_pos_name = "trainPos.txt",
                 train_neg_name = "trainNeg.txt", verif_name = "verification.txt"):
    train_pos = np.loadtxt(train_pos_name)
    train_neg = np.loadtxt(train_neg_name)
    verif = np.loadtxt(verif_name)
    return train_pos, train_neg, verif
def script_test():
    folder_statue = "./test_data/statue/"
    pcf_file = folder_statue + "Rekonstruktion30.pcf"
    imgL, imgR = scr.load_images(folder_statue)
    build_dataset(pcf_file, imgL, imgR)
script_test()
