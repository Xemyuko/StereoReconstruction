# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 21:04:55 2025

@author: myuey
"""
import scripts as scr
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import ndimage
import cv2

def plot_func():
    x = np.linspace(0,20,50)
    y=np.zeros(50)
    f=5
    y[x<f] = (-f*x[x<f]) / (f-x[x<f])
    y[x>f] = (-f*x[x>f]) / (f-x[x>f])
    pos = [12,13]
    x[pos] = np.nan
    y[pos] = np.nan
    plt.plot(x,y)
    plt.title("f = " + str(f))
    plt.xlabel("S1")
    plt.ylabel("S2")
    plt.axvline(x = 5, color = 'r')
    plt.show()

plot_func()
    
def salt_pepper_noise(img):
    # Getting the dimensions of the image
    row , col, color = img.shape
    
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
      
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
        
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
        
        # Color that pixel to white
        img[y_coord][x_coord] = 255
        
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
      
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
        
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
        
        # Color that pixel to black
        img[y_coord][x_coord] = 0
        
    return img
    
def medfil_display():
    #load image
    filepath = "C:/Users/myuey/Documents/250912_denoise2/gearref/cam1_pos_0000pattern_0000.jpg"
    img = cv2.imread(filepath)
    plt.imshow(img)
    plt.show()
    #create noisy image
    noise_img = salt_pepper_noise(img)
    plt.imshow(noise_img)
    plt.show()
    #apply median filter
    med_fil_img = ndimage.median_filter(noise_img, size=15)
    plt.imshow(med_fil_img)
    plt.show()
    
