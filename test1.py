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


def gradient_img():
    img_path = './test_data/denoise_unet/eval_target/cam1_pos_0010pattern_0000.jpg'
    img = cv2.imread(img_path)
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
 
    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
 
    plt.show()

    
def p1():
    x = np.linspace(-2,2,50)
    y1 = 2*np.ones_like(x) 
    y2 = 2*x
    y3 = x*x
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,y3)
    plt.show()
    
def p3d():
    x = np.arange(-5, 5, 0.25)
    y = np.arange(-5, 5, 0.25)
    x, y = np.meshgrid(x, y)
    z = np.sin(x)*np.cos(y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, cmap=plt.cm.jet,
                       linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.set_ylabel('y')
    ax.set_xlabel('x')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def p2():
    x = np.arange(-5, 5, 0.25)
    y = np.sin(x)
    
    plt.plot(x,y)
    plt.xlabel('x, y = 0')
    plt.ylabel('z')
    plt.show()
    
p2()

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
    

    
