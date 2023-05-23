# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 20:54:59 2023

@author: myuey
"""
import numpy as np
from tqdm import tqdm
import scripts as scr
from scipy.interpolate import LinearNDInterpolator
import random
import matplotlib.pyplot as plt
float_chk = 1e-9
card_ind = [0,1,0,1,1,0,1,0]
def generate_neighbors(yC, xC, yLim, xLim):
    x = [-1,-1,-1,0,0,1,1,1]
    y = [-1,0,1,-1,1,-1,0,1]
    res = []
    for i,j in zip(y,x):
        pN = np.asarray([int(yC)+i,int(xC)+j])
        if(pN[0] >= 0 and pN[0] < yLim and pN[1] >= 0 and pN[1]< xLim):
            res.append(pN)
        else:
            res.append(np.asarray([-1,-1]))
    return np.asarray(res, dtype = 'int32')
def access_data(img_stack, yC, xC, yLim, xLim):
    val = []
    if (xC < 0 and yC < 0):
        val = np.zeros((30,))
    elif(xC - int(xC) <= float_chk and yC - int(yC) <= float_chk):
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

def scramble_data(scram, ref):
    train_neg = []
    total_len = len(scram) + len(ref)
    for i in range(len(scram)):
        rand_range = list(set((range(0,total_len, 1))) - set([i]))
        rand_val = random.choice(rand_range)
        scram_pair = []
        if(rand_val < len(scram)):
            scram_pair = scram[rand_val]
        else:
            scram_pair = ref[rand_val - len(scram)]
        neg_entry = scram[i]
        neg_entry[1] = scram_pair[1]
        train_neg.append(neg_entry)
    return np.asarray(train_neg, dtype = 'float32')

def split_pairing_data(xyL,xyR,imgL, imgR, yLim, xLim):
    train_pos = [] #0
    train_scram = []#1
    verif_pos = [] #2
    verif_scram = []
    prev_code = -1
    counter_pos = 0
    flag_verif = True
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
        entry = []
        entry_data_c_L = np.asarray(entry_data_c_L)
        entry_data_c_R = np.asarray(entry_data_c_R)
        entry_data_n_L = np.asarray(entry_data_n_L)
        entry_data_n_R = np.asarray(entry_data_n_R)
        
        counter_L=0
        for dat_n,card in zip(entry_data_n_L, card_ind):
            
            if(counter_L == 4):
                entry.append(entry_data_c_L * 10)
            if(card == 0):
                entry.append(dat_n*2)
            else:
                entry.append(dat_n*4)
            counter_L+=1
        counter_R = 0
        for dat_n,card in zip(entry_data_n_R, card_ind):
            
            if(counter_R == 4):
                entry.append(entry_data_c_R * 10)
            if(card == 0):
                entry.append(dat_n*2)
            else:
                entry.append(dat_n*4)
            counter_R+=1
        entry = np.asarray(entry, dtype = 'float32')
        
        if prev_code == -1 or prev_code == 2: #beginning or verif was prev, load into pos train
            train_pos.append(entry)
            counter_pos +=1
            prev_code = 0
        elif(prev_code == 0 and counter_pos < 2): #pos train was prev, pos train quota not met, load into pos train
            train_pos.append(entry)
            counter_pos +=1
            prev_code = 0
        elif(prev_code == 0 and counter_pos >= 2):#pos train was prev, pos train quota met, load into scram train
            train_scram.append(entry)
            counter_pos = 0
            prev_code = 1
        elif(prev_code == 1):#scram train was prev, load into verif
            if flag_verif:
                verif_pos.append(entry)
                flag_verif = False
            else:
                verif_scram.append(entry)
                flag_verif = True
            prev_code = 2

            
    tn = scramble_data(train_scram, train_pos)
    tp = np.asarray(train_pos, dtype = 'float32')
    train = np.concatenate((tp,tn))
    train_labels = np.concatenate((np.ones((tp.shape[0],)),np.zeros((tn.shape[0],))))
    train_labels = train_labels.astype('int32')
    vn= scramble_data(verif_scram, verif_pos)
    vp = np.asarray(verif_pos, dtype = 'float32')
    verif = np.concatenate((vp, vn))
    verif_labels = np.concatenate((np.ones((vp.shape[0],)),np.zeros((vn.shape[0],))))
    verif_labels = verif_labels.astype('int32')
    return train, train_labels, verif, verif_labels
def count_subpixel(xyList):
    counter = 0
    for i in xyList:
        if(i[0] - int(i[0] > float_chk or i[1] - int(i[1]) > float_chk)):
            counter+=1
    return counter
train_name = "train.npy"
train_lbl_name = "train_labels.npy"
verif_name = "verif.npy"
verif_lbl_name = "verif_labels.npy"
def build_dataset(pcf_file, imgL, imgR, yLim, xLim,inc_num = 100):
    
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(pcf_file)
    xy1 = xy1[::inc_num]
    xy2 = xy2[::inc_num]
    train, train_lbls, verif, verif_lbls = split_pairing_data(xy1, xy2, imgL, imgR, yLim, xLim)
    
    np.save(train_name, train)
    np.save(verif_name, verif)
    np.save(train_lbl_name, train_lbls)
    np.save(verif_lbl_name, verif_lbls)
def load_data(data_name, label_name):
    data = np.load(data_name, allow_pickle = True)
    labels = np.load(label_name, allow_pickle = True)
    return data,labels
def visualize_data_point(data, labels, ind):
    data_entry = data[ind]
    print("Shape: " + str(data_entry.shape) + " Label: "+str(bool(labels[ind])))
    plt.imshow(data_entry)
    plt.show()
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
    a, b = load_data(train_name, train_lbl_name)
    c, d = load_data(verif_name, verif_lbl_name)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)
    visualize_data_point(a,b, 500)
script_test()