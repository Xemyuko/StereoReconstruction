'''
Created on Nov 6, 2022

@author: myuey
'''
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from stereo_rectification import loop_zhang as lz
import json
import numba
from numba import cuda as cu
#used for comparing floating point numbers to avoid numerical errors
float_epsilon = 1e-9


def get_gpu_name():
    '''
    Gets GPU device name if it exists, if not, returns None
    '''
    res = None
    try:
        res = str(cu.current_context().device.name)[2:-1]
    except(Exception):
        res = None
    return res


    

def create_plane_pts(dist_scale, plane_triplet, plane_length_count):
    '''
    Creates a large number of points in a plane in space.

    Parameters
    ----------
    dist_scale : float
        distance between points
    plane_triplet : list of 3D float lists
        3 3D points defining the plane
    plane_length_count : integer
        DESCRIPTION.

    Returns
    -------
    res_pts : numpy array of 3D float arrays
        points in plane

    '''
    p0, p1, p2 = plane_triplet
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    ux, uy, uz = [x1-x0, y1-y0, z1-z0]
    vx, vy, vz = [x2-x0, y2-y0, z2-z0]

    u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

    point  = np.array(p0)
    normal = np.array(u_cross_v)

    d = -point.dot(normal)
    res_pts = []
    for i in range(-int(plane_length_count/2),int(plane_length_count/2)):
        for j in range(-int(plane_length_count/2),int(plane_length_count/2)):
            xx = i*dist_scale + plane_triplet[0][0] 
            yy = j*dist_scale + plane_triplet[0][1] 
            z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
            pt_entry = [xx,yy,z]
            res_pts.append(pt_entry)
    return np.asarray(res_pts)
def distance3D(pt1,pt2):
    '''
    Calculates 3D Euclidean distance between 2 3D points

    Parameters
    ----------
    pt1 : 3d iterable float
        first 3d point
    pt2 : 3d iterable float
        second 3d point

    Returns
    -------
    res : float
        3D Euclidean distance

    '''
    res = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)
    return res
def load_json_freeCAD(filename):
    '''
    Loads FreeCAD JSON export and extracts vertex positions

    Parameters
    ----------
    filename : String
        filepath of JSON file

    Returns
    -------
    vertices data
        list of 3d float lists

    '''
    f = open(filename)
    data = json.load(f)
    
    f.close()
    
    return data['objects'][0]['vertices']
def load_mats(folder, kL_file = "kL.txt", 
                 kR_file = "kR.txt", R_file = "R.txt", 
                 t_file = "t.txt",skiprow = 2, delim = " "):
    '''
    Loads camera constant matrices and related data from text files. 


    Parameters
    ----------
    folder : string
        Folder that matrices are stored in, ending in '/'.

    Returns
    -------
    kL : np array of shape (3,3), float
        left camera matrix.
    kR : np array of shape (3,3), float
        right camera matrix.
    r : np array of shape (3,3), float
        rotation matrix between cameras.
    t : np array of shape (3,1), float
        translation vector between cameras.

    '''
    kL = np.loadtxt(folder + kL_file, skiprows=skiprow, delimiter = delim)
    kR = np.loadtxt(folder + kR_file, skiprows=skiprow, delimiter = delim)
    r= np.loadtxt(folder + R_file, skiprows=skiprow, delimiter = delim)
    t= np.loadtxt(folder + t_file, skiprows=skiprow, delimiter = delim)
    return kL, kR, r, t
def create_data_out(ptsL, ptsR, cor, geom, col, filename1):
    '''
    Creates the datafile output of the program in txt format

    Parameters
    ----------
    ptsL : Iterable of 2D points
        Left view 2D points, ints
    ptsR : Iterable of 2D points
        Right view 2D points, ints
    cor : Iterable of correlation values
        Correlation values are floats or ints
    geom : Iterable of 3D points
        Floats
    col : Iterable of RGB colors
        Colors are 8bit integers
    filename1 : String
        Name of text file


    '''
    if "." in filename1:
        filename1 = filename1.split(".",1)[0]
    file_check = filename1 + ".txt"  
    counter = 1
    while os.path.exists(file_check):
        file_check = filename1 +"(" +str(counter)+")" + ".txt"
        counter += 1
    with open(file_check, 'w') as ori:  
        ori.write("'x1', 'y1', 'x2', 'y2', 'corr', 'x', 'y', 'z', 'r', 'g', 'b'\n")
        for i in range(len(ptsL)):
            ori.write(str(ptsL[i][0]) + " " + str(ptsL[i][1]) + " " + str(ptsR[i][0]) + " " + str(ptsR[i][1])+
                      " " + str(cor[i]) + " " + str(geom[i][0]) + " " + str(geom[i][1]) + " " + str(geom[i][2]) +
                      " " + str(col[i][0]) + " " + str(col[i][1]) + " " + str(col[i][2]) + "\n")
        ori.close()   
def create_pcf(xy1 , xy2, cor, geom, col, filename):
    '''
    Creates pcf data file with the following qualities:
    column names=['x1', 'y1', 'x2', 'y2', 'c', 'x', 'y', 'z', 'r', 'g', 'b']
    header: PCF1.0
            B0 H1 P1 C1 N0
    Expected inputs: xy1 , xy2, cor, geom, col
    '''

   
   
    header0 ='PCF1.0\n'
    header1 = 'B0 H1 P1 C1 N0\n'
    pcf_out= open(filename, "w")
    pcf_out.write(header0)
    pcf_out.write(header1)
    for i in range(geom.shape[0]):
        line = str(xy1[i][0]) + ' ' + str(xy1[i][1]) + ' ' + str(xy2[i][0]) + ' ' + str(xy2[i][1]) + ' ' + str(cor[i]) + ' '  + str(geom[i][0])+ ' ' + str(geom[i][1]) + ' ' + str(geom[i][2]) + ' ' + str(col[i][0])+ ' ' + str(col[i][1]) + ' ' + str(col[i][2])  + '\n'
        pcf_out.write(line)
    pcf_out.close()
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
def load_color_split(folderL = "",folderR = "", ext = ""):
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
    Left and right numpy arrays of images split into their RGB color channels.

    '''
    imgL = []
    imgR= [] 
    resL = []
    resR = []
    for file in os.listdir(folderL):
        if file.endswith(ext):
            resL.append(file)
    resL.sort()
    for i in resL:
        
        img = plt.imread(folderL + i)
        imgL.append(img[:,:,0])
        imgL.append(img[:,:,1])
        imgL.append(img[:,:,2])
    for file in os.listdir(folderR):
        if file.endswith(ext):
            resR.append(file)  
    resR.sort()
    for i in resR:
        img = plt.imread(folderR + i)
        imgR.append(img[:,:,0])
        imgR.append(img[:,:,1])
        imgR.append(img[:,:,2])
    return np.asarray(imgL),np.asarray(imgR)
def load_images_basic(folder, ext = ''):
    '''
    Loads all images with the given extension in the given folder
    '''
    res = []
    for file in os.listdir(folder):
        if file.endswith(ext):
            res.append(file)
    res.sort()
    img_list = []
    for a in res:
        img = plt.imread(folder + a)
        img_list.append(img)
    return img_list
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
def check_balance_1_dir(folder, imgLInd, imgRInd, ext):
    '''
    Checks if the number of images in a folder are equally split into left and right camera sections

    Parameters
    ----------
    folder : String
        DESCRIPTION.
    imgLInd : String
        DESCRIPTION.
    imgRInd : String
        DESCRIPTION.
    ext : String
        DESCRIPTION.

    Returns
    -------
    Boolean
        True if the folder is unbalanced or if one or both of the indicator strings are not found. 

    '''
    resL = []
    resR = []
    #Access and store all files with the image extension given
    imgFull = []
    for file in os.listdir(folder):
        if file.endswith(ext):
            imgFull.append(file)
    #Sort images into left and right based on if they contain the respective indicators
    #if they do not have either, ignore them
    for i in imgFull:
        if imgLInd in i:
            resL.append(i)
        elif imgRInd in i:
            resR.append(i)
    return len(resL) != len(resR) or len(resL) == 0 or len(resR) == 0
def load_images_1_dir(folder, imgLInd, imgRInd, ext = "", colorIm = False):
    '''
    Loads images from 1 directory using imgLInd and imgRInd to distinguish which image comes from which camera side. colorIm controls if the resulting images are in color. 
    '''
    imgL = []
    imgR = [] 
    resL = []
    resR = []
    #Access and store all files with the image extension given
    imgFull = []
    for file in os.listdir(folder):
        if file.endswith(ext):
            imgFull.append(file)
    #Sort images into left and right based on if they contain the respective indicators
    #if they do not have either, ignore them
     
    for i in imgFull:
        if imgLInd in i:
            resL.append(i)
        elif imgRInd in i:
            resR.append(i)      
    #sort left and right images
    resL.sort()
    resR.sort()
    for i in resL:
        img = plt.imread(folder + i)
        if len(img.shape) > 2 and not colorIm:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgL.append(img)
    for i in resR:
        img = plt.imread(folder + i)
        if len(img.shape) > 2 and not colorIm:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgR.append(img)
    return np.asarray(imgL),np.asarray(imgR)
def load_first_pair_1_dir(folder,imgLInd, imgRInd, ext):
    '''
    Loads first image pair left and right from the same folder with the given extension
    '''
    resL = []
    resR = []
    #Access and store all files with the image extension given
    imgFull = []
    for file in os.listdir(folder):
        if file.endswith(ext):
            imgFull.append(file)
    #Sort images into left and right based on if they contain the respective indicators
    #if they do not have either, ignore them
    for i in imgFull:
        if imgLInd in i:
            resL.append(i)
        elif imgRInd in i:
            resR.append(i)
    if len(resL) != len(resR) or len(resL) == 0 or len(resR) == 0:
        return np.asarray([]), np.asarray([])
    resL.sort()
    resR.sort()
    imgL = plt.imread(folder + resL[0])
    if len(imgL.shape) > 2:
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = plt.imread(folder + resR[0])
    if len(imgR.shape) > 2:
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    return imgL,imgR
def load_first_pair(folderL = "",folderR = "", ext = ""):
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
    img1 : numpy uint8 array
        First image in folderL
    img2 : numpy uint8 array
        First image in folderR

    '''
    resL = []
    resR = []
    for file in os.listdir(folderL):
        if file.endswith(ext):
            resL.append(file)
    resL.sort()
    
    for file in os.listdir(folderR):
        if file.endswith(ext):
            resR.append(file)
    resR.sort()
    img1 = plt.imread(folderL + resL[0])
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = plt.imread(folderR + resR[0])
    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1,img2

def create_ply(geo, file_name = 'testing.ply', overwrite = True):
    '''
    Creates a point cloud .ply file from a numpy array of 3d floats 

    Parameters
    ----------
    geo : numpy array of 3d floats
        Geometry list of points to convert to point cloud
    file_name : String, optional
        File name of the point cloud file. The default is 'testing.ply'.
    overwrite : Boolean, optional
        Controls if existing files with the name file_name should be overwritten. The default is True.

    Returns
    -------
    None.

    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(geo)
    pcd.colors = o3d.utility.Vector3dVector(gen_color_arr_black(geo.shape[0]))
    if "." in file_name:
        file_name = file_name.split(".",1)[0]
    file_check = file_name + ".ply"
    if (not overwrite):
        counter = 1
        while os.path.exists(file_check):
            file_check = file_name +"(" +str(counter)+")" + ".ply"
            counter += 1
    
    o3d.io.write_point_cloud(file_check, pcd)
def convert_np_ply(geo,col,file_name, overwrite = False):
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
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(geo)
    pcd.colors = o3d.utility.Vector3dVector(col)
    if "." in file_name:
        file_name = file_name.split(".",1)[0]
    file_check = file_name + ".ply"
    if (not overwrite):
        counter = 1
        while os.path.exists(file_check):
            file_check = file_name +"(" +str(counter)+")" + ".ply"
            counter += 1
    
    o3d.io.write_point_cloud(file_check, pcd)
def write_img(img, file_name):
    '''
    Creates a png image and avoids overwriting existing images with the same name.

    Parameters
    ----------
    img : numpy array
        image to be written. Can be grayscale or in color. 
    file_name : String
        Name of image file to be written
    Returns
    -------
    None.

    '''
    if "." in file_name:
        file_name = file_name.split(".",1)[0]
    file_check = file_name + ".png"  
    counter = 1
    while os.path.exists(file_check):
        file_check = file_name +"(" +str(counter)+")" + ".png"
        
        counter += 1
    cv2.imwrite(file_check, img)
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

def create_stereo_offset_fig(img1,img2,xOffsetL,xOffsetR,yOffsetT,yOffsetB):
    '''
    Creates figure for display in stereo with the specified offsets

    Parameters
    ----------
    img1 : numpy uint8 image array
        Left image
    img2 : numpy uint8 image array
        Right image
    xOffsetL : int
        Left x offset value
    xOffsetR : int
        Right x offset value
    yOffsetT : int
        Top y offset value
    yOffsetB : int
        Bottom y offset value

    Returns
    -------
    f : matplotlib figure
        Stereo image figure

    '''
    color1 = (255,0,0)
    imshape = img1.shape
    xLim = imshape[1]
    yLim = imshape[0]
    #convert images to color by stacking 3x
    img1 = np.stack((img1,img1,img1),axis = 2)
    img2 = np.stack((img2,img2,img2),axis = 2)
    thickness = 20
    img1 = cv2.rectangle(img1, (xOffsetL,yOffsetT), (xLim - xOffsetR,yLim - yOffsetB), color1,thickness) 
    img2 = cv2.rectangle(img2, (xOffsetL,yOffsetT), (xLim - xOffsetR,yLim - yOffsetB), color1,thickness) 
    f = plt.figure()
    f.set_figwidth(60)
    f.set_figheight(40)
    f.add_subplot(1,2,1)
    plt.imshow(img1, cmap = "gray")
    f.add_subplot(1,2,2)
    plt.imshow(img2, cmap = "gray")
    return f        
def mark_points(img1,img2,pts1,pts2,xOffset = 1,yOffset =1, size = 5, showBox = True):
    '''
    Marks points from lists onto images, with an optional box around the target 
    window of the pictures.

    Parameters
    ----------
    img1 : np array
        Left image
    img2 : np array
        Right image
    pts1 : list of np arrays
        List of left side points
    pts2 : list of np arrays
        List of right side points
    xOffset : int
        Distance in pixels offset from edge for x axis of window
    yOffset : int
        Distance in pixels offset from edge for y axis of window
    size : int, optional
        Diameter in pixels of points. The default is 5.
    showBox : Boolean, optional
        Controls display of window borders. The default is True.

    Returns
    -------
    None.

    '''
    print("# POINTS: " + str(len(pts1)))
    
    #place points
    for pt1,pt2 in zip(pts1,pts2):

        pt1 = np.round(pt1)
        pt2 = np.round(pt2)
        color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        img1 = cv2.circle(img1,tuple(pt1),size,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),size,color,-1)

                
            
    if(showBox):
        #place window boundaries    
        color1 = (255,0,0)
        imshape = img1.shape
        xLim = imshape[1]
        yLim = imshape[0]
        img1 = cv2.rectangle(img1, (xOffset,yOffset), (xLim - xOffset,yLim - yOffset), color1,1) 
        img2 = cv2.rectangle(img2, (xOffset,yOffset), (xLim - xOffset,yLim - yOffset), color1,1) 
    
    plt.subplot(121),plt.imshow(img1)
    plt.subplot(122),plt.imshow(img2)
    plt.show()
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()
    

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

    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
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
@numba.jit(nopython=True)
def ncc_f_mat_point_search(Gi, agi, val_i, imgs2, im_shape, n):
    '''
    Helper function to apply GPU acceleration
    '''
    max_cor = 0.0
    match_loc_x = 0.0
    match_loc_y = 0.0
    for a in range(im_shape[0]):
        for b in range(im_shape[1]):
            Gt = imgs2[:,a,b]
            agt = np.sum(Gt)/n        
            val_t = np.sum((Gt-agt)**2)
            if(val_i > float_epsilon and val_t > float_epsilon): 
                cor = np.sum((Gi-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))  
                if cor > max_cor:
                    max_cor = cor
                    match_loc_y = a
                    match_loc_x = b
    return max_cor, [match_loc_y,match_loc_x]



def find_f_mat_ncc(imgs1,imgs2, thresh = 0.7, f_calc_mode = 0, ret_pts = False):
    im_shape = imgs1[0].shape
    n = imgs1.shape[0]
    
    pair_list = []
    for i in range(0,im_shape[0],100):
        for j in range(0,im_shape[1],100):
            Gi = imgs1[:,i,j]
            agi = np.sum(Gi)/n
            val_i = np.sum((Gi-agi)**2)
            
            if(np.sum(Gi) != 0):
               max_cor, match_pt = ncc_f_mat_point_search(Gi, agi, val_i, imgs2, im_shape, n)
               if(max_cor > thresh):
                   add_flag = True
                   b = [i,j,match_pt[0],match_pt[1],max_cor]
                   for a in pair_list:
                       if(a[0] == b[0] and a[1] == b[1]):
                           add_flag = b[4] > a[4]
                           print(add_flag)
                   if add_flag:
                       pair_list.append(b)
          
    pair_list = np.asarray(pair_list)


    pts1 = []
    pts2 = []
    for i in pair_list:
        pts1.append([i[0],i[1]])
        pts2.append([i[2],i[3]]) 
           
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    F = None
    try:
        if(f_calc_mode == 0):
            
            F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
        elif(f_calc_mode == 1):
            F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)
        else:
            F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
    except(Exception):
      
        print("Failed to find fundamental matrix, likely due to insufficient input data.")
    if(ret_pts):
        return F,pts1,pts2
    else:
        return F
    
                                
def find_f_mat(img1,img2, thresh = 0.7, f_calc_mode = 0, ret_pts = False):
    '''
    Finds fundamental matrix using feature correlation.

    Parameters
    ----------
    img1 : numpy uint8 image array
        Left image
    img2 : numpy uint8 image array
        Right image
    thresh : float, optional
        Threshold for feature matching. The default is 0.8.
    lmeds_mode : boolean, optional
        Pick if LMEDS or 8-point algorithms are used. The default is True.

    Returns
    -------
    F : 3x3 numpy float array matrix
        Fundamental matrix correlating the two images.

    '''
    
    
    
    #identify feature points to correlate
    sift = cv2.SIFT_create()
    pts1 = []
    pts2 = []
    sp1, des1 = sift.detectAndCompute(img1,None)
    sp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    for i,(m,n) in enumerate(matches):
        if m.distance < thresh*n.distance:
            pts2.append(sp2[m.trainIdx].pt)
            pts1.append(sp1[m.queryIdx].pt)
            
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    
    
    F = None
    try:
        if(f_calc_mode == 0):
            
            F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
        elif(f_calc_mode == 1):
            F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)
        else:
            F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
    except(Exception):
        print("Failed to find fundamental matrix, likely due to insufficient input data.")
    if(ret_pts):
        return F,pts1,pts2
    else:
        return F

def find_f_mat_list(im1,im2,thresh = 0.7, f_calc_mode = 0, ret_pts = False):
    
    pts1v = []
    pts2v = []
    counter = 0
    while len(pts1v) < 10 and counter < len(im1):
        pts1 = []
        pts2 = []
        img1 = im1[counter]
        img2 = im2[counter]
        counter+=1
        #identify feature points to correlate
        sift = cv2.SIFT_create()
        sp1, des1 = sift.detectAndCompute(img1,None)
        sp2, des2 = sift.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        for i,(m,n) in enumerate(matches):
            if m.distance < thresh*n.distance:
                pts2.append(sp2[m.trainIdx].pt)
                pts1.append(sp1[m.queryIdx].pt)

    
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2) 
        
        for i,j in zip(pts1,pts2):
            Gi = im1[:,i[1],i[0]]
            agi = np.sum(Gi)/len(im1)
            val_i = np.sum((Gi-agi)**2)
            if(np.sum(Gi) != 0):
                Gt = im2[:,j[1],j[0]]
                agt = np.sum(Gt)/len(im1)
                val_t = np.sum((Gt-agt)**2)
                cor = None
                if(val_i > float_epsilon and val_t > float_epsilon): 
                    cor = np.sum((Gi-agi)*(Gt - agt))/(np.sqrt(val_i*val_t))
                if cor != None and cor >= thresh:
                    pts1v.append(i)
                    pts2v.append(j)

    pts1v = np.asarray(pts1v)
    pts2v = np.asarray(pts2v)

    F = None
    try:
        if(f_calc_mode == 0):
            
            F, mask = cv2.findFundamentalMat(pts1v,pts2v,cv2.FM_LMEDS)
        elif(f_calc_mode == 1):
            F, mask = cv2.findFundamentalMat(pts1v,pts2v,cv2.FM_8POINT)
        else:
            F, mask = cv2.findFundamentalMat(pts1v,pts2v,cv2.FM_RANSAC)
    except(Exception):
        print("Failed to find fundamental matrix, likely due to insufficient input data.")
        
    if(ret_pts):
        return F,pts1v,pts2v
    else:
        return F
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
def display_4_comp(img1,img2,img3,img4):
    f = plt.figure()
    f.add_subplot(2,2,1)
    plt.imshow(img1, cmap = "gray")
    f.add_subplot(2,2,2)
    plt.imshow(img2, cmap = "gray")
    f.add_subplot(2,2,3)
    plt.imshow(img3, cmap = "gray")
    f.add_subplot(2,2,4)
    plt.imshow(img4, cmap = "gray")
    plt.show()
def pair_list_corr(img_listL,img_listR, color = False, thresh = 0.8):
    '''
    

    Parameters
    ----------
    img_listL : list of np arrays
        list of images for left side
    
    img_listR : list of np arrays
        list of images for right side
    
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

    '''
    pts1 = []
    pts2 = []
    col_res = []
    for i,j in tqdm(zip(img_listL,img_listR)):
        res1,res2,col, F = feature_corr(i,j, color, thresh)
        for a,b,c in zip(res1,res2,col):
            pts1.append(a)
            pts2.append(b)
            col_res.append(c)
    return pts1,pts2,col_res
def triangulate(pt1,pt2,R,t,kL,kR):
    '''
    Main triangulation function. Finds 3D position of point in real space from camera left and right 2D points.

    Parameters
    ----------
    pt1 : 2D float iterable
        Left image point
    pt2 : 2D float iterable
        Right image point
    R : 3x3 np float array 
        Rotation matrix between cameras
    t : 3x1 np float array
        Translation vector between cameras
    kL : 3x3 np float array 
        Left camera matrix
    kR : 3x3 np float array 
        Right camera matrix

    Returns
    -------
    3x1 np float array 
        3D point in real space

    '''
    #Create calc matrices 
    Al = np.c_[kL, np.asarray([[0],[0],[0]])]


    RT = np.c_[R, t]
    Ar = kR @ RT

    sol0 = pt1[1] * Al[2,:] - Al[1,:]
    sol1 = -pt1[0] * Al[2,:] + Al[0,:]
    sol2 = pt2[1] * Ar[2,:] - Ar[1,:]
    sol3 = -pt2[0] * Ar[2,:] + Ar[0,:]
    
    solMat = np.stack((sol0,sol1,sol2,sol3))
    #Apply SVD to solution matrix to find triangulation
    U,s,vh = np.linalg.svd(solMat,full_matrices = True)

    Q = vh[3,:]

    Q /= Q[3]


    return Q[0:3]

def triangulate_list_nobar(pts1, pts2, r_vec, t_vec, kL, kR):
    '''

    Applies the triangulate function to all points in a list without a progress bar. 

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
    for i in range(len(pts1)):
        res.append(triangulate(pts1[i],pts2[i],r_vec, t_vec, kL, kR))
    return np.asarray(res)
def triangulate_list(pts1, pts2, r_vec, t_vec, kL, kR):
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
    for i in tqdm(range(len(pts1))):
        res.append(triangulate(pts1[i],pts2[i],r_vec, t_vec, kL, kR))
    return np.asarray(res)
def bin_convert_arr(img_arr, val):
    '''
    Converts images in an array into binary images based on intensity threshold, then returns them in array format. 

    Parameters
    ----------
    img_arr : numpy array
        array of images to be converted
    val : int or float
        intensity threshold for binary conversion

    Returns
    -------
    numpy array
        Array of converted images

    '''
    res_list = []
    for i in range(img_arr.shape[0]):
        res = np.zeros_like(img_arr[i])
        res[img_arr[i]>val] = 1
        res = res.astype(np.float)
        res_list.append(res)
    return np.asarray(res_list)   
def multi_bin_convert_list(imgList,vals, conv_type = np.int32):
    '''
    Converts images in list to lists of binary images 
    based on a list of threshold values
        
    Parameters
    ----------
    imgList : list of np arrays
        list of images to convert to binary
    vals : list of ints
        list of threshold values to check against
    conv_type : type of resulting images. Defaults to np.int32

    Returns
    -------
    bin_list : list of np arrays
        list of binary images. Can be 3 dimensional if vals input has multiple entries

    '''
    bin_list = []
    for i in imgList:
        res_entry = []
        for j in vals:
            res = np.zeros_like(i)
            res[i>j] = 1
            res = res.astype(conv_type)
            res_entry.append(res)
        bin_list.append(res_entry)
    return bin_list

def disp_map(image_shape,pts1,pts2):
    '''
    Generates disparity map from points and result shape

    Parameters
    ----------
    image_shape : Tuple 
        shape of image to map
    pts1 : list of np arrays
        left image points
    pts2 : list of np arrays
        right image points

    Returns
    -------
    res : np array
        Disparity map image

    '''
    res = np.zeros(image_shape)
    for i,j in zip(pts1,pts2):
        dist = int(abs(j[0] - i[0]))
        res[int(i[1]),int(i[0])] = dist
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

def mask_img(img,thresh):
    '''
    Masks an image to a given threshold.

    Parameters
    ----------
    img : numpy image array
        image to be masked
    thresh : float or int, matching dtype of img
        Threshold to mask at. 

    Returns
    -------
    mask_img: 
        DESCRIPTION.

    '''
    mask = np.ones_like(img)
    mask[img < thresh] = 0
    return img*mask
def mask_inten_list(img_list,thresh_val):
    '''
    Applies mask value to list of images

    Parameters
    ----------
    img_list : List of numpy image arrays
        List of images
    thresh_val : int or float
        Mask Threshold value
    Returns
    -------
    res_list : List of numpy image arrays
        Masked image list

    '''
    res_list = []
    for i in img_list:
        mask = np.ones_like(i)
        mask[i < thresh_val] = 0
        res_list.append(i*mask)
    return res_list
def crop_img(img, xOffsetL, xOffsetR, yOffsetT, yOffsetB):
    '''
    Crops image according to the 4 offsets.

    Parameters
    ----------
    img : Numpy image array
        Image to be cropped
    xOffsetL : int
        left x offset value
    xOffsetR : int
        right x offset value
    yOffsetT : int
        top y offset value
    yOffsetB : int
        bottom y offset value

    Returns
    -------
    numpy array
        cropped image

    '''
    return img[yOffsetT:img.shape[1]-yOffsetB,xOffsetL:img.shape[0]-xOffsetR]

def boost_zone(img, scale_factor,xOffsetL, xOffsetR, yOffsetT, yOffsetB):
    '''
    Increases values of pixels in an image within the input offsets area according to the scale factor.
    Scales down the result to maintain the same data type as the input image. 

    Parameters
    ----------
    img : numpy array
        image to be modified 
    scale_factor : int or float
        factor to multiply image values by
    xOffsetL : int
        left x offset value
    xOffsetR : int
        right x offset value
    yOffsetT : int
        top y offset value
    yOffsetB : int
        bottom y offset value

    Returns
    -------
    res : numpy array, same dtype as img
        resulting image with boosted zone

    '''
    res = np.copy(img)
    roi = img[yOffsetT:img.shape[1]-yOffsetB,xOffsetL:img.shape[0]-xOffsetR]
    roi2 = roi*float(scale_factor)
    try:
        roi2 = np.clip(roi2, 0, np.iinfo(roi.dtype).max).astype(roi.dtype)
    except(ValueError):
        roi2 = np.clip(roi2, 0, np.finfo(roi.dtype).max).astype(roi.dtype)
    
    
    res[yOffsetT:img.shape[1]-yOffsetB,xOffsetL:img.shape[0]-xOffsetR] = roi2
    return res
def boost_list(img_list, scale_factor,xOffsetL, xOffsetR, yOffsetT, yOffsetB):
    '''
    Applies the boost_zone function to a list of images, with the same zone for each image in the list

    Parameters
    ----------
    img_list : list of numpy arrays
        list of images
    scale_factor : int or float
        factor to multiply image values by
    xOffsetL : int
        left x offset value
    xOffsetR : int
        right x offset value
    yOffsetT : int
        top y offset value
    yOffsetB : int
        bottom y offset value
    Returns
    -------
    res : TYPE
        DESCRIPTION.

    '''
    res = []
    for i in img_list:
        res.append(boost_zone(i, scale_factor,xOffsetL, xOffsetR, yOffsetT, yOffsetB))
    return res
def mask_avg_list(avg_img, img_list, thresh_val):
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
 
@numba.jit(nopython=True)   
def col_help(lims, images, i, thresh, res_red, res_red_count, res_green, res_green_count, res_blue, res_blue_count):
    for j in range(lims[1]):
        val_stack = images[:,i,j,:]
        for a in range(val_stack.shape[0]):
            r_val = val_stack[a,0]
            if r_val > thresh:
                res_red[i,j] += r_val
                res_red_count[i,j] += 1
            g_val = val_stack[a,1]
            if g_val > thresh:
                res_green[i,j] += g_val
                res_green_count[i,j] += 1
            b_val = val_stack[a,2]
            if b_val > thresh:
                res_blue[i,j] += b_val
                res_blue_count[i,j] += 1
    return res_red, res_green, res_blue, res_red_count, res_green_count, res_blue_count         
def get_color(imagesL,imagesR,ptsL,ptsR, mode = 1):
    #create 7 empty arrays of same shape as image, 3 to store running sums of each channel, 3 to store count of values added, 1 for result
    res_imageL = np.zeros(imagesL[0].shape)
    res_redL = np.zeros(imagesL[0,:,:,0].shape)
    res_red_countL = np.ones(imagesL[0,:,:,0].shape)
    res_blueL = np.zeros(imagesL[0,:,:,0].shape)
    res_blue_countL = np.ones(imagesL[0,:,:,0].shape)
    res_greenL = np.zeros(imagesL[0,:,:,0].shape)
    res_green_countL = np.ones(imagesL[0,:,:,0].shape)
    
    res_imageR = np.zeros(imagesR[0].shape)
    res_redR = np.zeros(imagesR[0,:,:,0].shape)
    res_red_countR = np.ones(imagesR[0,:,:,0].shape)
    res_blueR = np.zeros(imagesR[0,:,:,0].shape)
    res_blue_countR = np.ones(imagesR[0,:,:,0].shape)
    res_greenR = np.zeros(imagesR[0,:,:,0].shape)
    res_green_countR = np.ones(imagesR[0,:,:,0].shape)
    #establish color intensity thresholds for rejection of value
    thresh = 10
    lims = imagesL[0].shape
    #loop through stack of images 3xn and retrieve all 3 color channels for each pixel for each image
    for i in range(lims[0]):
        res_redL, res_greenL, res_blueL, res_red_countL, res_green_countL, res_blue_countL = col_help(lims, imagesL, i, thresh, res_redL, res_red_countL, res_greenL, res_green_countL, res_blueL, res_blue_countL)
        res_redR, res_greenR, res_blueR, res_red_countR, res_green_countR, res_blue_countR = col_help(lims, imagesR, i, thresh, res_redR, res_red_countR, res_greenR, res_green_countR, res_blueR, res_blue_countR)
    res_imageL[:,:,0] = res_redL/res_red_countL/255
    res_imageL[:,:,1] = res_greenL/res_green_countL/255
    res_imageL[:,:,2] = res_blueL/res_blue_countL/255 
    
    res_imageR[:,:,0] = res_redR/res_red_countR/255
    res_imageR[:,:,1] = res_greenR/res_green_countR/255
    res_imageR[:,:,2] = res_blueR/res_blue_countR/255 

    
    res_col = []

    for a in range(len(ptsL)):
        try:
            if(mode == 0):
                res_col.append((res_imageL[ptsL[a][0],ptsL[a][1],:]))
            else:
                res_col.append((res_imageL[ptsL[a][1],ptsL[a][0],:]))
        except:
            try:
                if(mode == 0):
                    res_col.append((res_imageR[ptsR[a][0],ptsR[a][1],:]))
                else:
                    res_col.append((res_imageR[ptsR[a][1],ptsR[a][0],:]))
            except:
                res_col.append(np.asarray([0,0,0]))

    res_col = np.asarray(res_col)
    return res_col

def get_color_1_pair(imL,imR, ptsL, ptsR):
    res_col = []
    
    for a in range(len(ptsL)):
        try:
            if np.max((imL[ptsL[a][1],ptsL[a][0],:])) > 0:
                res_col.append((imL[ptsL[a][1],ptsL[a][0],:]/255))
            else:
                res_col.append((imL[ptsL[a][1],ptsL[a][0],:]))
        except:
            try:
                if np.max((imR[ptsR[a][1],ptsR[a][0],:])) > 0:
                    res_col.append((imR[ptsR[a][1],ptsR[a][0],:]/255))
                else:
                    res_col.append((imR[ptsR[a][1],ptsR[a][0],:]))
            except:
                res_col.append(np.asarray([0,0,0]))
    res_col = np.asarray(res_col)
    return res_col
@numba.jit(nopython=True)
def accel_dist_count(data, data_point, data_ind, thresh_dist, thresh_count):
    '''
    GPU accelerated distance checker between selected point and rest of data. Counts number of neighbours.

    Parameters
    ----------
    data : numpy array of 3d float iterables
        3D point data to be checked against
    data_point : 3d float iterable
        3D point to compare against data
    data_ind : integer
        Location of data_point in data to avoid comparing aginst itself
    thresh_dist : float
        distance threshold to count a point pair as neighbouring
    thresh_count : integer
        number of neighbours threshold 

    Returns
    -------
    counter : integer
        number of neighbours of data_point in data

    '''
    counter = 0
    for i in range(data.shape[0]):
        if i != data_ind:
            p1 = data[i]
            p2=data_point
            dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
            if(dist < thresh_dist):
                counter+=1
                if counter > thresh_count:
                    break
    return counter
def cleanup(data,dist_val,noise_scale, thresh_count, outlier_scale):
    '''
    Cleans up 3D point data by removing outliers and noise.

    Parameters
    ----------
    data : numpy array of 3d float arrays
        data to be modified
    dist_val : float
        distance scaling value indicating how far away points are from each other
    noise_scale : integer
        number of dist_val away from a point that should be considered a neighbor
    thresh_count : integer
        number of neighbours that are needed to not be considered noise.
    outlier_scale : integer
        number of dist_val away from centroid that should be considered outliers

    Returns
    -------
    numpy array of 3d float arrays
        cleaned 3D point data

    '''
    #Run code for removing outliers on data and distance threshold
    centroid = np.asarray([np.mean(data[:,0]), np.mean(data[:,1]),np.mean(data[:,2])]) 
    res_arr_out = []
    for i in range(data.shape[0]):
        out2 = data[i,:]
        dist = np.sqrt((centroid[0]-out2[0])**2 + (centroid[1]-out2[1])**2 + (centroid[2]-out2[2])**2)
        if dist < outlier_scale * dist_val:
            res_arr_out.append(out2)
    res_arr_out = np.asarray(res_arr_out)
    #Run code for removing noise on result of above
    res = []
    for i in range(res_arr_out.shape[0]):
        counter = 0
        counter = accel_dist_count(res_arr_out, res_arr_out[i,:], i, noise_scale * dist_val, thresh_count)
        if counter > thresh_count:
            res.append(res_arr_out[i,:])
    #Return cleaned data
    return np.asarray(res)
def gen_color_arr_black(pts_len):
    '''
    Generates numpy array of length given of black RGB values.

    Parameters
    ----------
    pts_len : int
        number of black values to create

    Returns
    -------
    res : numpy float32 array
        array of black RGB color values

    '''
    res = []
    for i in range(pts_len):
        val = np.asarray([0.0,0.0,0.0])
        res.append(val)
    return np.asarray(res, dtype = np.float32)
def get_pix_stack(imgs1,imgs2,x1,y1, x2,y2):
    '''
    

    Parameters
    ----------
    imgs1 : List of Numpy arrays
        first list of images
    imgs2 : List of Numpy arrays
        second list of images
    x1 : int
        x location in list 1
    y1 : int
        y location in list 1
    x2 : int
        x location in list 2
    y2 : int
        y location in list 2

    Returns
    -------
    stack1 : numpy array
        first stack of pixels
    stack2 : numpy array
        second stack of pixels

    '''
    stack1 = []
    stack2 = []
    for i,j in zip(imgs1,imgs2):
        stack1.append(i[y1][x1])
        stack2.append(j[y2][x2])
    stack1 = np.asarray(stack1)
    stack2 = np.asarray(stack2)
    return stack1,stack2
def conv_rect_map_list(disp_map, HL, HR):
    '''
    Converts a rectified image disparity map into a pair of 
    point lists for the two original images that were used to create the map
    
    Parameters
    ----------
    disp_map : np array
        grayscale image (single intensity value) indicating point disparities
    HL : np array of shape (3,3)
        Rectification matrix for left image
    HR : TYPE
        Rectification matrix for right image

    Returns
    -------
    ptsL : np array
        3D points with third dimension very close to 1 for left image
    ptsR : np array
        3D points with third dimension very close to 1 for right image

    '''
    imshape = disp_map.shape
    pts1 = []
    pts2 = []
    for i in range(imshape[0]):
        for j in range(imshape[1]):
            val = disp_map[i,j]
            pts1.append(np.asarray([i,j]))
            pts2.append(np.asarray([i+val,j]))
    ptsL = []
    ptsR = []
    hL_inv = np.linalg.inv(HL)
    hR_inv = np.linalg.inv(HR)
    for a,b in zip(pts1,pts2):
        sL = HL[2,0]*a[0] + HL[2,1] * a[1] + HL[2,2]
        pL = hL_inv @ np.asarray([[a[0]],[a[1]],[sL]])
        sR = HR[2,0]*b[0] + HR[2,1] * b[1] + HR[2,2]
        pR = hR_inv @ np.asarray([[b[0]],[b[1]],[sR]])
        ptsL.append([pL[0,0],pL[1,0],pL[2,0]])
        ptsR.append([pR[0,0],pR[1,0],pR[2,0]])
    return ptsL,ptsR
def load_all_imgs_1_dir(folder, ext = "",convert_gray = False):
    '''
    Loads images from a single directory in alphanumerical order.

    Parameters
    ----------
    folder : String
        directory images are in
    ext : String
        optional extension of images of interest. 
        If not provided, all images will be loaded.

    Returns
    -------
    image_list : List of numpy arrays
        List of images loaded from directory.

    '''
    image_list = []
    res = []
    
    for file in os.listdir(folder):
        if file.endswith(ext):
            res.append(file)
    res.sort()
    for i in res:
        img = cv2.imread(folder + i)
        if(convert_gray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_list.append(img)
    return image_list
def fill_mtx_dir(folder, kL, kR, fund, ess, distL, distR, R, t):
    '''
    
    Writes camera calibration matrices to a folder. If the folder does not exist, creates the folder. 
    
    Parameters
    ----------
    folder : String
        folder to write to
    kL : numpy array
        left camera intrinsic matrix
    kR : numpy array
        right camera intrinsic matrix
    fund : numpy array
        fundamental matrix
    ess : numpy array
        essential matrix
    distL : numpy array
        left camera distortion
    distR : numpy array
        right camera distortion
    R : numpy array
        Rotation matrix
    t : numpy array
        Translation vector

    Returns
    -------
    None.

    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savetxt(folder + "kL.txt", kL, header = "3\n3")
    np.savetxt(folder + "kR.txt", kR, header = "3\n3")
    np.savetxt(folder + "f.txt", fund, header = "3\n3")
    np.savetxt(folder + "e.txt", ess, header = "3\n3")
    np.savetxt(folder + "distL.txt", distL, header = "1\n3")
    np.savetxt(folder + "distR.txt", distR, header = "1\n3")
    np.savetxt(folder + "R.txt", R, header = "3\n3")
    np.savetxt(folder + "t.txt", t, header = "1\n3")
def calibrate_single(images, ext, rows, columns, world_scaling):
    '''
    Calibrates a single camera, generating a distortion vector and a camera matrix

    Parameters
    ----------
    images : list of numpy arrays
        calibration grid images
    ext : string
        image file extension
    rows : int
        Number of rows in calibration chessboard grid
    columns : int
        Number of columns in calibration chessboard grid
    world_scaling : float
        length of edges in calibration chessboard grid in meters

    Returns
    -------
    mtx : 3x3 numpy array
        Camera intrinsic values matrix
    dist : numpy array vector
        Camera distortion coefficients

    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
    width = images[0].shape[1]
    height = images[0].shape[0]
    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
   # chkfrm_list = []
    for i in tqdm(range(len(images))):
        frame = images[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_thr = int(gray.max()*0.8)
        mask1 = np.ones_like(gray)
        mask1[gray < mask_thr] = 0 
        gray = gray*mask1
        #find the checkerboard
        
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), cv2.CALIB_CB_ADAPTIVE_THRESH)
        
        if ret == True:
            
            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)
 
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
           # checkframe = cv2.drawChessboardCorners(frame, (rows,columns), corners, ret)
           # chkfrm_list.append(checkframe)
            objpoints.append(objp)
            imgpoints.append(corners)
    print("Resolving Calibration...")    
    try:    
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    except Exception as e:
        print(e)
        print("Calibration Failure.")
        mtx = None
        dist = None
    return mtx,dist


    

    
def calibrate_cameras(cal_folder, left_mark, right_mark, ext, rows, columns, world_scaling):
    '''
    Calibrates the stereo cameras. If this process fails, all return values are None.

    Parameters
    ----------
    kL_folder : String
        folder containing left camera calibration images
    kR_folder : String
        folder containing right camera calibration images
    ext : String
        extension of images in folders
    rows : int
        number of rows in calibration chessboard
    columns : int
        number of columns in calibration chessboard
    world_scaling : float
        Real distance of sides in calibration chessboard, in meters

    Returns
    -------
    mtx1: numpy 3x3 array
        Left camera matrix
    mtx2: numpy 3x3 array
        Right camera matrix
    dist_1: numpy array
        Left camera distortion coefficients
    dist_2: numpy array
        Right camera distortion coefficients
    R: 3x3 numpy array
        Camera system rotation matrix
    T: numpy array
        Camera system translation vector
    E: 3x3 numpy array
        Camera system essential matrix
    F: 3x3 numpy array
        Camera system fundamental matrix

    '''
    print('Loading Calibration Images')
    #load images from each folder in numerical order
    images1, images2 = load_images_1_dir(cal_folder, left_mark,right_mark, ext, colorIm = True)
    
    
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
    #frame dimensions. Frames should be the same size.
    width = images1[0].shape[1]
    height = images1[0].shape[0]
    
    #Apply the opencv camera calibration function  to get kL, kR, R, and t
    #calibrate single cameras to get matrices and distortions
    print("Calibrating Left Camera")
    mtx1, dist_1 = calibrate_single(images1, ext, rows, columns, world_scaling)
    if mtx1 is not None:
        print("Calibrating Right Camera")
        mtx2, dist_2 = calibrate_single(images2, ext, rows, columns, world_scaling)
        if mtx2 is not None:
            #Pixel coordinates of checkerboards
            imgpoints_left = [] # 2d points in image plane.
            imgpoints_right = []
            objpoints = []
            for frame1, frame2 in zip(images1, images2):
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                c_ret1, corners1 = cv2.findChessboardCorners(gray1, (rows,columns), None)
                c_ret2, corners2 = cv2.findChessboardCorners(gray2, (rows,columns), None)
 
                if c_ret1 == True and c_ret2 == True:
                    corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                    corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
                
                    #cv.drawChessboardCorners(frame1, (rows,columns), corners1, c_ret1)
                    #cv.imshow('img', frame1)
 
                    #cv.drawChessboardCorners(frame2, (rows,columns), corners2, c_ret2)
                    #cv.imshow('img2', frame2)
                    #k = cv.waitKey(0)
 
                    objpoints.append(objp)
                    imgpoints_left.append(corners1)
                    imgpoints_right.append(corners2)
    
    
            stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
            print("Running Stereo Calibration...")
            ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist_1,
                                                                 mtx2, dist_2, (width, height), criteria = criteria, flags = stereocalibration_flags)
            print("Calibration Complete.")
            return mtx1, mtx2, dist_1, dist_2, R, T, E, F
    return None, None, None, None, None, None, None, None

def undistort(images, mtx, dist):
    '''
    Undistorts images based on distortion coefficients and calculates a new camera matrix. 

    Parameters
    ----------
    images : list of numpy arrays
        list of images taken with distorted camera lens
    mtx : numpy array
        camera matrix with distortion
    dist : numpy array
        distortion coefficients array

    Returns
    -------
    new_mtx : numpy array of floats
        undistorted camera matrix
    images_res : list of numpy arrays
        undistorted images

    '''
    img_dim = images[0]
    ho,wo = img_dim.shape
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (wo,ho), 1, (wo,ho))
    images_res = []
    for img in images:
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, new_mtx)
        # crop the image
        x, y, w, h = roi
        dst = np.asarray(dst[y:y+h, x:x+w])
        images_res.append(dst)
    return new_mtx, images_res

def pcf_to_ply(pcf_loc, target_ply):
    xy1,xy2,geom_arr,col_arr,correl = read_pcf(pcf_loc)
    convert_np_ply(geom_arr, col_arr, target_ply)

def corr_calibrate(pts1,pts2, kL, kR, F):
    '''
    Finds rotation matrix, and translation vector from matching points, intrinsic camera matrices, and fundamental matrix. 

    '''
    ess = kR.T @ F @ kL
    #a,R,t,b = cv2.recoverPose(ess,pts1,pts2)
   # t=t.T[0]
    #return R,t

    s = cv2.decomposeEssentialMat(ess)
    print(s)
    R1 = s[0]
    R2 = s[1]
    t = s[2]
    a = triangulate(pts1[0],pts2[0], R1,t, kL, kR)


    if a[2] > 0:
        
        return R1,t
    else:
        return R2,t 
    
