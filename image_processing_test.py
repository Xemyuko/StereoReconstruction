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

    img_blur = cv2.GaussianBlur(img,(3,3), sigmaX=0, sigmaY=0) 

    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) 
 
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)
    
edge_detect()