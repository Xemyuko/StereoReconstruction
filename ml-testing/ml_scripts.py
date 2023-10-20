# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:47:09 2023

@author: Admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

def read_pcf(inputfile):
    '''
    Reads a .pcf file with the column names=['x1', 'y1', 'x2', 'y2', 'c', 'x', 'y', 'z', 'r', 'g', 'b']
    and the data separated by spaces with the first two rows being header information/blank
    Parameters
    ----------
    inputfile : TYPE
        File path of pcf file.

    Returns
    -------
    xy1 : numpy array, float64
        x and y coordinates for the first camera, pulled from x1 and y1 columns.
    xy2 : numpy array, float64
        x and y coordinates for the second camera, , pulled from x2 and y2 columns.
    geom_arr : numpy array
        3D points, pulled from x,y,z columns. 
    col_arr : numpy array
        color values in RGB space, pulled from r,g,b columns.
    correl : numpy array
        correlation values pulled from column c. 
    '''
    df  = pd.read_table(inputfile, skiprows=2, sep = " ", names=['x1', 'y1', 'x2', 'y2', 'c', 'x', 'y', 'z', 'r', 'g', 'b'])
    geom = df[['x','y','z']]
    col = df[['r','g','b']]
    geom_arr = geom.to_numpy()
    col_arr = col.to_numpy()
    
    xy1 = df[['x1','y1']]
    xy2 = df[['x2','y2']]
    correl = df['c']
    xy1 = xy1.to_numpy()
    xy1 = np.asarray(xy1, 'float64')
    xy2 = xy2.to_numpy()
    xy2 = np.asarray(xy2, 'float64')
    correl = correl.to_numpy()
    return xy1,xy2,geom_arr,col_arr,correl

def display_stereo(img1,img2):
    '''
    Displays two images in a stereo figure

    Parameters
    ----------
    img1 : Numpy image array
        Left image
    img2 : Numpy image array
        Right image

    Returns
    -------
    None.

    '''
    f = plt.figure()
    f.add_subplot(1,2,1)
    plt.imshow(img1, cmap = "gray")
    f.add_subplot(1,2,2)
    plt.imshow(img2, cmap = "gray")
    plt.show()
    
def load_images(folderL = "",folderR = "", ext = ""):
    '''
    

     Parameters
     ----------
     folderL : String, optional
         Left image folder. The default is "".
     folderR : String, optional
         Right image folder. The default is "".
     ext : TYPE, optional
         Image file extension. The default is "".

     Returns
     -------
     Left and right numpy arrays of images.


    '''
    imgL = []
    imgR = [] 
    resL = []
    resR = []
    for file in os.listdir(folderL):
        if file.endswith(ext):
            resL.append(file)
    resL.sort()
    for i in resL:
        img = plt.imread(folderL + i)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgL.append(img)     
    for file in os.listdir(folderR):
        if file.endswith(ext):
            resR.append(file)
    resR.sort()
    for i in resR:
        img = plt.imread(folderR + i)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgR.append(img)   
    return np.asarray(imgL),np.asarray(imgR)
def normal_pad(image_list):
    #find max shape of images in list
    
    
    pass