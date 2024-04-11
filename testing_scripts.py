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
import time 
import threading as thr
from numba import cuda as cu
import time
import os
import matplotlib.pyplot as plt
import confighandler as chand
from scipy.interpolate import Rbf
import scipy.interpolate
import tkinter as tk
import multiprocess as mupr
#used for comparing floating point numbers to avoid numerical errors
float_epsilon = 1e-9

class StoppableThread(thr.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = thr.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

def test_multhr():
    # Create Object 
    root = tk.Tk() 
  
    # Set geometry 
    root.geometry("400x400") 
    global t1 
    t1= None
    # use threading 
   
    def threading(): 
        # Call work function
        global t1
        t1=StoppableThread(target=work) 
        t1.start() 
    def stop():
        global t1 
        print('STOP')
        if(t1 != None):
            t1.stop()
            
            print('CANCELLED')
    # work function 
    def work(): 
  
        print("sleep time start") 
  
        for i in range(10): 
            print(i) 
            time.sleep(1) 
  
        print("sleep time stop") 
    def on_close():
        stop()
        root.destroy()
    # Create Button 
    tk.Button(root,text="Start",command = threading).pack() 
    tk.Button(root,text="Cancel",command = stop).pack() 
    # Execute Tkinter 
    root.mainloop()
test_multhr()

def test_mupr():
    root = tk.Tk()
    root.geometry("400x400")
    global proc
    
    def mpr():
        proc = mupr.Process(target=work, args=())
        proc.start()
    def mpr_trm():
        proc.terminate()
    def work(): 
  
        print("sleep time start") 
  
        for i in range(10): 
            print(i) 
            time.sleep(1) 
  
        print("sleep time stop")
    tk.Button(root,text="Start",command = mpr).pack() 
    tk.Button(root,text="Cancel",command = mpr_trm).pack() 
    root.mainloop()



def conv_pcf_ply():
    pcf_loc = './test_data/testset0/240312_angel/000POS000Rekonstruktion030.pcf'
    out_file = 'ref_angel.ply'
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
    
    res = cv2.recoverPose(xy1[:20].T,xy2[:20].T,E)
    
    print(res)
    

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
def plotSP(x,y,z,grid):
    plt.figure()
    plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()))
    plt.scatter(x,y,c=z, edgecolors = 'white')
    plt.colorbar()
    
def test_interp0():
    # Setup: Generate data
    n = 10
    nx, ny = 50, 50
    x, y, z = map(np.random.random, [n, n, n])
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    xi, yi = np.meshgrid(xi, yi)
    xi, yi = xi.flatten(), yi.flatten()

    grid3 = linear_rbf(x,y,z,xi,yi)
    grid3 = grid3.reshape((ny, nx))

    plotSP(x,y,z,grid3)
    plt.title('linear Rbf')

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

def test_interp1():
    #More close testing to actual use case
    #Generate data - 8-neighbor + Central point = 9 points known for x,y,z
    #Set x and y locations
    #Set z values as well for consistency, but add option to randomize
    x_val = np.asarray([0,0.5,1,0,0.5,1,0,0.5,1])
    y_val = np.asarray([0,0,0,0.5,0.5,0.5,1,1,1])
    z_val = np.asarray([1,0,1,0.5,0.5,0.5,0,1,0])
    randZ = False

    if randZ:
        z_val = np.random.rand(9)
    n = 11
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
    print(grid[1,1])
    plotSP(x_val,y_val,z_val,grid)
    plt.title('linear Rbf')

    plt.show()
    '''
    print(z_val)
    
    a = 0.6
    b = 0.5
    #search interp field returns incorrect results when looking for known points used to create the interpolation field
    c = search_interp_field(grid,a,b,x_val.min(), x_val.max(),y_val.min(), y_val.max(), n)
    print(c)
    '''
    print('############')




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
    
    

    n = 3
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

@numba.jit(nopython=True) 
def test_interp_numba():
    x_val = np.asarray([0,0.5,1,0,0.5,1,0,0.5,1])
    y_val = np.asarray([0,0,0,0.5,0.5,0.5,1,1,1])
    z_val = np.asarray([1,0,1,0.5,0.5,0.5,0,1,0])
    randZ = False

    if randZ:
        z_val = np.random.rand(9)
    n = 11
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
    weights = np.linalg.solve(internal_dist, z_val)
    grid =  np.dot(dist.T, weights)
    grid = grid.reshape((n, n))
    print(grid[1,1])


def remove_z_outlier(geom_arr, col_arr):
    ind_del = []
    for i in range(geom_arr.shape[0]):

        if geom_arr[i,2] > np.max(geom_arr[:,2])*0.8:
            ind_del.append(i)

    geom_arr = np.delete(geom_arr, np.asarray(ind_del), axis = 0)
    col_arr = np.delete(col_arr,np.asarray(ind_del), axis = 0 )
    return geom_arr, col_arr

def remove_z_outlier_no_col(geom_arr):
    ind_del = []
    for i in range(geom_arr.shape[0]):

        if geom_arr[i,2] > np.max(geom_arr[:,2])*0.8:
            ind_del.append(i)

    geom_arr = np.delete(geom_arr, np.asarray(ind_del), axis = 0)

    return geom_arr

def test_fix2():
    #Load Matrices
    testFolder = "./test_data/maustest/"
    input_data = "Maus1.pcf"
    skiprow = 2
    delim = ' '
    kL = np.loadtxt(testFolder + 'Kl.txt', skiprows=skiprow, delimiter = delim)
    kR = np.loadtxt(testFolder +  'Kr.txt', skiprows=skiprow, delimiter = delim)
    kL_inv = np.linalg.inv(kL)
    kR_inv = np.linalg.inv(kR)
    r_vec = np.loadtxt(testFolder +  'R.txt', skiprows=skiprow, delimiter = delim)
    t_vec = np.loadtxt(testFolder +  't.txt', skiprows=skiprow, delimiter = delim)
    #Access 2D points from reference pcf
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(testFolder + input_data)
    res=[]
    for i,j in zip(xy1,xy2):
        res.append(scr.triangulate_avg(i,j,r_vec,t_vec, kL_inv, kR_inv ))
    res = np.asarray(res)
    scr.create_ply(res, "testmaus2")

def test_fix():
    #Load Matrices
    testFolder = "./test_data/maustest/"
    input_data = "Maus1.pcf"
    skiprow = 2
    delim = ' '
    kL = np.loadtxt(testFolder + 'Kl.txt', skiprows=skiprow, delimiter = delim)
    kR = np.loadtxt(testFolder +  'Kr.txt', skiprows=skiprow, delimiter = delim)
    
    r_vec = np.loadtxt(testFolder +  'R.txt', skiprows=skiprow, delimiter = delim)
    t_vec = np.loadtxt(testFolder +  't.txt', skiprows=skiprow, delimiter = delim)
    #Create calc matrices 

    k1 = np.c_[kL, np.asarray([[0],[0],[1]])]
    
    k2 = np.c_[kR, np.asarray([[0],[0],[1]])]
    
    RT = np.c_[r_vec, t_vec]
    RT = np.r_[RT, [np.asarray([0,0,0,1])]]
    
    P1 = k1 @ np.eye(4,4)
    
    P2 = k2 @ RT
    #Access 2D points from reference pcf
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(testFolder + input_data)
    res=[]
    cor_thresh = 0.6
    ignore_thresh = True
    for i,j,k in zip(xy1,xy2, correl):
        #Check correlation threshold
        
        if(k >= cor_thresh or ignore_thresh):
            #Create solution matrix
            sol0 = i[0] * P1[2,:] - P1[0,:]
            sol1 = i[1] * P1[2,:] - P1[1,:]
            sol2 = j[0] * P2[2,:] - P2[0,:]
            sol3 = j[1] * P2[2,:] - P2[1,:]
            
            solMat = np.stack((sol0,sol1,sol2,sol3))
            
            #Apply SVD to solution matrix to find triangulation
            U,s,vh = np.linalg.svd(solMat,full_matrices = True)
            vh = vh.T
            Q = vh[:,3]

            Q *= 1/Q[3]
    
            res.append(Q[:3])
    res = np.asarray(res)
    filt_res = []
    filt_thresh = 0.6
    for i,j in zip(geom_arr,correl):
        if(j >= filt_thresh):
            filt_res.append(i)
    filt_res = np.asarray(filt_res)
    scr.create_ply(res, "testmaus")
    scr.create_ply(geom_arr, "referencemaus")
    scr.create_ply(filt_res, 'filtmaus')

def t1():
    folder = "./test_data/calibObjects/"
    filename = folder + 'testconeread.json'
    f = open(filename)
    data = json.load(f)
    vertices = np.asarray(data['objects'][0]['vertices'])
    f.close()
    maxVals = np.max(vertices, axis = 0)
    minVals = np.min(vertices, axis = 0)
    refXdist = maxVals[0] - minVals[0]
    refYdist = maxVals[2] - minVals[2]
    refZdist = maxVals[1] - minVals[1]
    chkZdist = 40
    print(data['objects'][0])
    


    
