# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:27:05 2023

@author: Admin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_detect():
    file_loc = './test_data/calibObjects/coneobjectpic.png'
    img = cv2.imread(file_loc,flags=0)  
    plt.imshow(img)
    plt.show()
    img_blur = cv2.GaussianBlur(img,(3,3), sigmaX=0, sigmaY=0) 

    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) 
    
    plt.imshow(edges)
    plt.show()
#edge_detect()

def vertex_detect():
    file_loc = './test_data/calibObjects/coneobjectpic.png'
    img = cv2.imread(file_loc,flags=0) 
    plt.imshow(img)
    plt.show()
    gray = np.float32(img)
    imgcol = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    corners = cv2.goodFeaturesToTrack(gray,20,0.01,10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(imgcol,(x,y),3,255,-1)
    plt.imshow(imgcol)
    plt.show()
    
vertex_detect()