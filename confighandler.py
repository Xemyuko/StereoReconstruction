# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 17:49:34 2023

@author: myuey
"""
import os
class ConfigHandler():
    
    def __init__(self, version_num):
        self.version = version_num
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
        self.tmod =  0.416657633#13
        self.config_filename = "recon_config.txt"
        self.mask_thresh = 30 #14
        self.output = "recon.ply" #15
        self.f_file = "fund.txt" #16
        self.f_load = 0 #17
        self.f_save = 0 #18
        self.precise = 0 #19
        self.speed_mode = 0 #20
        self.speed_interval = 10 #21
        self.corr_map_name = "correlation_map" #22
        self.left_calib = "calib_left/"  
        self.right_calib = "calib_right/" 
        self.calib_rows = 8 
        self.calib_columns = 12 
        self.calib_len = 0.04 
        
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
        config_file.write(str(self.x_offset) + "\n")
        config_file.write(str(self.y_offset) + "\n")
        config_file.write(str(self.interp) + "\n")
        config_file.write(str(self.thresh) + "\n")
        config_file.write(str(self.tmod)+ "\n")
        config_file.write(str(self.mask_thresh) + "\n")
        config_file.write(self.output + "\n")
        config_file.write(self.f_file + "\n")
        config_file.write(str(self.f_load)+ "\n")
        config_file.write(str(self.f_save)+ "\n")
        config_file.write(str(self.precise)+ "\n")
        config_file.write(str(self.speed_mode)+ "\n")
        config_file.write(str(self.speed_interval)+ "\n")
        config_file.write(self.corr_map_name + "\n")
        config_file.close()
        
    def load_config(self):
        if os.path.isfile(self.config_filename):  
            config_file = open(self.config_filename, "r")
            res = config_file.readlines()
            try:
                self.mat_folder = res[0][:-1]
                self.kL_file = res[1][:-1]
                self.kR_file = res[2][:-1]
                self.t_file = res[3][:-1]
                self.R_file = res[4][:-1]
                self.skiprow = int(res[5][:-1])
                self.delim = res[6][:-1]
                self.left_folder = res[7][:-1]
                self.right_folder = res[8][:-1]
                self.x_offset = int(res[9][:-1])
                self.y_offset = int(res[10][:-1])
                self.interp = int(res[11][:-1])
                self.thresh = float(res[12][:-1])
                self.tmod = float(res[13][:-1])
                self.mask_thresh = int(res[14][:-1])
                self.output = res[15][:-1]
                self.f_file = res[16][:-1]
                self.f_load = int(res[17][:-1])
                self.f_save = int(res[18][:-1])
                self.precise = int(res[19][:-1])
                self.speed_mode = int(res[20][:-1])
                self.speed_interval = int(res[21][:-1])
                self.corr_map_name = res[22][:-1]
            except(ValueError, IndexError):
                print("Invalid values found in existing configuration file, rebuilding configuration file with default values.")
                self.make_config()
        else:
            self.make_config()
