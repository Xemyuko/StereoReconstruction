# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 17:49:34 2023

@author: myuey
"""
class ConfigHandler():
    
    def __init__(self):
        self.mat_folder = "matrix_folder/" #0
        self.kL_file = "kL.txt" #1
        self.kR_file = "kR.txt" #2
        self.t_file = "t.txt" #3
        self.R_file = "R.txt" #4
        self.skiprow = 2 #5
        self.delim = " " #6
        self.left_folder = "camera_L/" #7
        self.right_folder = "camera_R/" #8
        self.x_offset = 1 #9
        self.y_offset = 1 #10
        self.interp = 3 #11
        self.thresh = 0.9 #12
        self.tmod =  0.583342367 #13
        self.config_filename = "config.txt"
        
    def make_config(self):
        config_file = open(self.config_filename, "w")
        config_file.write(self.mat_folder + "\n")
        config_file.write(self.kL_file + "\n")
        config_file.write(self.kR_file + "\n")
        config_file.write(self.t_file + "\n")
        config_file.write(self.R_file + "\n")
        config_file.write(str(self.skiprow) + "\n")
        config_file.write(self.delim + "\n")
        config_file.write(self.left_folder + "\n")
        config_file.write(self.right_folder + "\n")
        config_file.write(str(self.xOff) + "\n")
        config_file.write(str(self.yOff) + "\n")
        config_file.write(str(self.interp) + "\n")
        config_file.write(str(self.thresh) + "\n")
        config_file.write(str(self.tmod))
        config_file.close()
        
    def load_config(self):
        config_file = open(self.config_filename, "r")
        res = config_file.readlines()
        res_clean = []
        self.mat_folder = res[0][:-1]
        res_clean.append(self.mat_folder)
        self.kL_file = res[1][:-1]
        res_clean.append(self.kL_file)
        self.kR_file = res[2][:-1]
        res_clean.append(self.kR_file)
        self.t_file = res[3][:-1]
        res_clean.append(self.t_file)
        self.R_file = res[4][:-1]
        res_clean.append(self.R_file)
        self.skiprow = int(res[5][:-1])
        res_clean.append(self.skiprow)
        self.delim = res[6][:-1]
        res_clean.append(self.delim)
        self.left_folder = res[7][:-1]
        res_clean.append(self.left_folder)
        self.right_folder = res[8][:-1]
        res_clean.append(self.right_folder)
        self.x_offset = int(res[9][:-1])
        res_clean.append(self.x_offset)
        self.y_offset = int(res[10][:-1])
        res_clean.append(self.y_offset)
        self.interp = int(res[11][:-1])
        res_clean.append(self.interp)
        self.thresh = float(res[12][:-1])
        res_clean.append(self.thresh)
        self.tmod = float(res[13])
        res_clean.append(self.tmod)
        return res_clean