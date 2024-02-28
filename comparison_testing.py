# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 09:32:11 2024

@author: myuey
"""
import scripts as scr
import confighandler as cfh
import ncc_core as ncc



def compare_ply():
    pass
    #compare max X,Y,Z dimensions
    #compare relative XYZ positions of reconstructed points from the same pair of 2D points
    #compare distances between several reconstructed points from the same pairs of 2D points
    #compare centroids of point clouds
    #compare the minimum distance between 2 points

def test1_conversion():
    
    drive = "H:/"
    folder = "ball/pws/"
    filepath = drive + folder + "000POS1Rekonstruktion30.pcf"
    output_filename = "refball.ply"
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(filepath)
    scr.convert_np_ply(geom_arr, col_arr, output_filename, overwrite=True)
test1_conversion()

def test2_compare_same_points():
    
    drive = "E:/"
    folder = "Mice1Left/"
    folder_pcf = folder+"pws/"
    filepath = drive + folder_pcf + "000POS1Rekonstruktion30.pcf"
    xy1,xy2,geom_arr,col_arr,correl = scr.read_pcf(filepath)
    
    kL, kR, r, t = scr.load_mats(folder, kL_file = "Kl.txt", kR_file = "Kr.txt", R_file = "R.txt", 
                  t_file = "t.txt",skiprow = 2, delim = " ")
    geom_check = scr.triangulate_list(xy1, xy2, r, t, kL, kR)
    