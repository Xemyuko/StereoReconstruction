# -*- coding: utf-8 -*-
"""
Created on Tue May  2 19:15:44 2023

@author: myuey
"""
import cv2
from stereo_rectification import loop_zhang as lz
from matplotlib.pyplot import imread
import numpy as np
import os
from open3d import utility, geometry,io

def initial_load(tMod,folder, kL_file = "kL.txt", 
                 kR_file = "kR.txt", R_file = "R.txt", 
                 t_file = "t.txt",skiprow = 2, delim = " "):
    '''
    Loads camera constant matrices and related data from text files. 


    Parameters
    ----------
    tMod : float
        translation vector correction constant
    folder : string, optional
        Folder that matrices are stored in, ending in '/'.

    Returns
    -------
    kL : np array of shape (3,3), float
        left camera matrix.
    kR : np array of shape (3,3), float
        right camera matrix.
    r_vec : np array of shape (3,3), float
        rotation matrix between cameras.
    t_vec : np array of shape (3,1), float
        translation vector between cameras.

    '''
    kL = np.loadtxt(folder + kL_file, skiprows=skiprow, delimiter = delim)
    kR = np.loadtxt(folder + kR_file, skiprows=skiprow, delimiter = delim)
    r_vec = np.loadtxt(folder + R_file, skiprows=skiprow, delimiter = delim)
    t_vec = np.loadtxt(folder + t_file, skiprows=skiprow, delimiter = delim)
    
    t_vec = t_vec[:,np.newaxis]*tMod
    return kL, kR, r_vec, t_vec
def load_images(folderL = "",folderR = "", ext = ""):
    imgL = []
    imgR = [] 
    resL = []
    resR = []
    for file in os.listdir(folderL):
        if file.endswith(ext):
            resL.append(file)
    resL.sort()
    for i in resL:
        img = imread(folderL + i)
        imgL.append(img)     
    for file in os.listdir(folderR):
        if file.endswith(ext):
            resR.append(file)
    resR.sort()
    for i in resR:
        img = imread(folderR + i)
        imgR.append(img)   
    return np.asarray(imgL),np.asarray(imgR)
def convert_np_ply(geo,col,file_name):
    '''
    Converts geometry and color arrays into a .ply point cloud file. 

    Parameters
    ----------
    geo : numpy array
        3D geometry points data
    col : numpy array
        color values in RGB colorspace
    file_name : string
        Name for file path to be created. Adding ".ply" to the end is not needed. 

    Returns
    -------
    None.

    '''
    pcd = geometry.PointCloud()
    pcd.points = utility.Vector3dVector(geo)
    pcd.colors = utility.Vector3dVector(col)
    
    if "." in file_name:
        file_name = file_name.split(".",1)[0]
    file_check = file_name + ".ply"
    counter = 1
    while os.path.exists(file_check):
        file_check = file_name +"(" +str(counter)+")" + ".ply"
        counter += 1
    io.write_point_cloud(file_check, pcd)
def conv_pts(ptsList):
    '''
    Converts points from 3D to 2D by removing the 3rd entry.
    For use after unrectifying previously rectified points

    Parameters
    ----------
    ptsList : list of 3D points

    Returns
    -------
    res_list : list of 2D points
    '''

    res_list = []
    for i in ptsList:
        res_list.append([i[0],i[1]])
    return res_list
def feature_corr(img1,img2, color = False, thresh = 0.8):
    '''
    Applies SIFT feature detection and FLANN knn feature correlation to find pairs of matching points between two images

    Parameters
    ----------
    img1 : np array
        First image to search for matches in
    img2 : np array
        Second image to search for matches in
    color : Boolean, optional
        Boolean to control if the images are in color or grayscale (single intensity channel)
        The default is False.
    thresh : float, optional
        Threshold for two points to be considered a match. 
        Higher values will lead to more points, but more errors.  The default is 0.8.

    Returns
    -------
    pts1 : np array
        2D points from image 1
    pts2 : np array
        2D points from image 2
    col_vals : np array
        Color values for the matched points in RGB space, found by averaging the two matches.
        If image is grayscale and color argument is False, 
        the single intensity channel will be duplicated twice to shift it into RGB space
    F : np array of shape (3,3)
        fundamental matrix calculated from matching points using LMEDS algorithm      
    '''
    #identify feature points to correlate
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    sp1, des1 = sift.detectAndCompute(img1,None)
    sp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    pts1 = []
    pts2 = []

    for i,(m,n) in enumerate(matches):
        if m.distance < thresh*n.distance:
            pts2.append(sp2[m.trainIdx].pt)
            pts1.append(sp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)

    #Remove outliers

    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    pts_val1 = []
    pts_val2 = []
    for i in range(len(pts1)):
        p1 = pts1[i]
        p2 = pts2[i]
        s = img1.shape
        if not (p1[0] >= s[0] or p1[1] >= s[1] or p2[0] >= s[0] or p2[1] >= s[1]):
            pts_val1.append(p1)
            pts_val2.append(p2)
    pts1 = np.asarray(pts_val1)
    pts2 = np.asarray(pts_val2)

    #Copy point color values to array

    col_vals = []
    for i,j in zip(pts1,pts2):
        c1 = img1[i[0]][i[1]]/255
        c2 = img2[j[0]][j[1]]/255
        c_val = (c1+c2)/2
        col_vals.append(c_val)
    if (color):
        col_vals = np.asarray(col_vals)
    else:
        col_vals = np.asarray(col_vals)
        col_vals = np.column_stack((col_vals,col_vals,col_vals))
    
    
    col_vals = np.asarray(col_vals)
    return pts1,pts2,col_vals,F
  
def triangulate(pt1, pt2, r_vec, t_vec, kL_inv, kR_inv):
    '''
    Triangulates the 3D point in real space of 2 points in image space.

    Parameters
    ----------
    pt1 : np array/iterable
        Left 2D point
    pt2 : np array/iterable
        Right 2D point
    r_vec : np array of shape (3,3)
        rotation matrix between cameras
    t_vec : np array of shape (3,1)
        translation vector between cameras
    kL_inv : np array of shape (3,3), float
        Inverse left camera matrix.
    kR_inv : np array of shape (3,3), float
        Inverse right camera matrix.

    Returns
    -------
    res: np array
        3D point

    '''


    #Convert points to column vectors from row vectors and make them 3D

    p1 = np.asarray([[pt1[0]],[pt1[1]],[1]])
    p2 = np.asarray([[pt2[0]],[pt2[1]],[1]])
    #Take inverses of camera matrices and multiply points to transform image points into vectors from camera center
    #Apply transform for right camera points
    v1 = kL_inv @ p1
    v2 = r_vec@(kR_inv @ p2) + t_vec
    #Calculate distances along each vector for closest points, then sum and halve for midpoints

    
    phi = (t_vec[0,0]-v1[0,0]*t_vec[2,0])/(v1[0,0]*v2[2,0]-v2[0,0])
    
    lam = t_vec[2,0]+phi*v2[2,0]
    
    res = [(lam*v1[0,0]+phi*v2[0,0])/2,(lam*v1[1,0]+phi*v2[1,0])/2,(lam*v1[2,0]+phi*v2[2,0])/2]
    
    return np.asarray(res)

def triangulate_list(pts1, pts2, r_vec, t_vec, kL_inv, kR_inv):
    '''
    Applies the triangulate function to all points in a list.

    Parameters
    ----------
    pts1 : list of np arrays
        list of left points
    pts2 : list of np arrays 
        list of right points
    r_vec : np array of shape (3,3)
        rotation matrix between cameras
    t_vec : np array of shape (3,1)
        translation vector between cameras
    kL_inv : np array of shape (3,3), float
        Inverse left camera matrix.
    kR_inv : np array of shape (3,3), float
        Inverse right camera matrix.

    Returns
    -------
    res : np array
        3D points in array form for each pair of 2D points

    '''
    res = []
    for i,j in zip(pts1,pts2):
        res.append(triangulate(i,j,r_vec, t_vec, kL_inv, kR_inv))
    return res
def rectify_pair(imgL,imgR,F):
    '''
    Rectifies a pair of images using the Loop-Zhang algorithm. 
    
    Parameters
    ----------
    imgL : np array
        left image
    imgR : np array
        right image
    F : np array of shape (3,3)
        Fundamental matrix 

    Returns
    -------
    img1 : np array
        rectified left image 
    img2 : np array
        rectified right image
    H1 : np array of shape (3,3)
        Rectification matrix of left image 
    H2 : np array of shape (3,3)
        Rectification matrix of right image
    '''
    imshape = imgL.shape
    H1 = None
    H2 = None
    revshape = (imshape[1],imshape[0])
    H1, H2 = lz.stereo_rectify_uncalibrated(F, revshape)
    img1 = cv2.warpPerspective(imgL, H1, revshape)
    img2 = cv2.warpPerspective(imgR, H2, revshape)
    return img1,img2, H1, H2

def rectify_lists(imgL,imgR,F):
    '''
    Applies rectify_pair to paired lists of images

    Parameters
    ----------
    imgL : List of left images
 
    imgR : List of right images

    F : numpy array of shape (3,3)
        Fundamental matrix
 

    Returns
    -------
    res_listL : List of rectified left images
    res_listR : List of rectified right images

    '''
    res_listL = []
    res_listR = []
    for i,j in zip(imgL,imgR):
        res1,res2,a,b = rectify_pair(i,j, F)
        res_listL.append(res1)
        res_listR.append(res2)
    return res_listL, res_listR

def mask_inten_list(avg_img, img_list, thresh_val):
    '''
    Masks images in list based on a threshold. All regions where the average of the stack
    below the threshold will be set to 0. 

    Parameters
    ----------
    avg_img : np array
        Image holding average values across the stack
    img_list : list of np arrays
        list of images
    thresh_val : int
        threshold value to check against

    Returns
    -------
    res_list : list of np arrays
        list of masked images

    '''
    mask = np.ones_like(avg_img)
    mask[avg_img < thresh_val] = 0   
    res_list = []
    for i in img_list:
        res_list.append(i*mask)
    return res_list            

def gen_color_arr(ref_imageL, ref_imageR, ptsL, ptsR):
    '''
    Returns array of colors pulled from ref_images in the same order as the points in pts. 

    Parameters
    ----------
    ref_imageL and ref_imageR : uint8 array 
        Image data array
    ptsL and ptsR : integer array or list of arrays/tuples
        2D points
        
    Returns
    -------
    res: Numpy array of float RGB colors 

    '''
    res = []
    for i,j in zip(ptsL,ptsR):
        valL = np.asarray([0,0,0])
        valR = np.asarray([0,0,0])
        a = int(np.round(i[0]))
        b = int(np.round(i[1]))
        if(a<ref_imageL.shape[0] and b < ref_imageL.shape[1] and a > 0 and b > 0):
            col = ref_imageL[a,b]/255
            valL = np.asarray([col,col,col])
        c = int(np.round(j[0]))
        d = int(np.round(j[1]))
        if(c<ref_imageR.shape[0] and d < ref_imageR.shape[1] and c > 0 and d > 0):
            col = ref_imageR[c,d]/255
            valR = np.asarray([col,col,col]) 
        entry = np.mean([valL, valR], axis = 0)
        if entry[0] >= 1 or entry[0] < 0:
            entry = np.asarray([0.0,0.0,0.0])
        res.append(entry)
        
    return np.asarray(res)
