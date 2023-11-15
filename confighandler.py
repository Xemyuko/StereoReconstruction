# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 17:49:34 2023

@author: myuey
"""
import os
class ConfigHandler():
    '''
    Stores inputs and settings in a text file for later retrieval. 
    Also serves as an information storage object for passing these values between components of the program. 
    '''
    def __init__(self):
        '''
        Default values.
        '''
        self.mat_folder = "matrix_folder/" #0
        self.kL_file = "kL.txt" #1
        self.kR_file = "kR.txt" #2
        self.t_file = "t.txt" #3
        self.R_file = "R.txt" #4
        self.skiprow = 2 #5
        self.delim = " " #6
        self.left_folder = "camera_L/" #7
        self.right_folder = "camera_R/" #8
        self.x_offset_L = 1 #9
        self.x_offset_R = 1 #10
        self.y_offset_T = 1 #11
        self.y_offset_B = 1 #12
        self.interp = 3 #13
        self.thresh = 0.9 #14
        self.tmod =  0.416657633#15
        self.config_filename = "recon_config.txt"
        self.mask_thresh = 30 #16
        self.output = "recon.ply" #17
        self.f_file = "fund.txt" #18
        self.f_load = 0 #19
        self.f_save = 0 #20
        self.precise = 1 #21
        self.speed_mode = 0 #22
        self.speed_interval = 10 #23
        self.corr_map_name = "correlation_map.png" #24
        self.data_out = 0#25
        self.data_name = "corr_data.txt" #26
        self.corr_map_out = 0#27
        self.calib_left = "calib_left/" #28 
        self.calib_right = "calib_right/"  #29
        self.calib_target = "calib_mtx/" #30
        self.calib_rows = 8 #31
        self.calib_columns = 12 #32
        self.calib_scale = 0.04 #33
        self.data_xyz_name = "recon.xyz" #34
        self.ref_pcf = "reference.pcf" #35
        self.max_tmod = 1.0 #36
        self.color_recon = 0 #37 
    def make_config(self):
        '''
        Write self values to text file
        '''
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
        config_file.write(str(self.x_offset_L) + "\n")
        config_file.write(str(self.x_offset_R) + "\n")
        config_file.write(str(self.y_offset_T) + "\n")
        config_file.write(str(self.y_offset_B) + "\n")
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
        config_file.write(str(self.data_out) + "\n")
        config_file.write(self.data_name + "\n")
        config_file.write(str(self.corr_map_out) + "\n")
        config_file.write(self.calib_left + "\n")
        config_file.write(self.calib_right + "\n")
        config_file.write(self.calib_target + "\n")
        config_file.write(str(self.calib_rows) + "\n")
        config_file.write(str(self.calib_columns) + "\n")
        config_file.write(str(self.calib_scale) + "\n")
        config_file.write(self.data_xyz_name + "\n")
        config_file.write(self.ref_pcf + "\n")
        config_file.write(str(self.max_tmod) + "\n")
        config_file.write(str(self.color_recon) + "\n")
        config_file.close()
        
    def load_config(self):
        '''
        If config file exists, read it and store values. 
        If reading causes errors, replace existing file with new file using default values. 
        If no file exists, create new file using default values. 
        '''
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
                self.x_offset_L = int(res[9][:-1])
                self.x_offset_R = int(res[10][:-1])
                self.y_offset_T = int(res[11][:-1])
                self.y_offset_B = int(res[12][:-1])
                self.interp = int(res[13][:-1])
                self.thresh = float(res[14][:-1])
                self.tmod = float(res[15][:-1])
                self.mask_thresh = int(res[16][:-1])
                self.output = res[17][:-1]
                self.f_file = res[18][:-1]
                self.f_load = int(res[19][:-1])
                self.f_save = int(res[20][:-1])
                self.precise = int(res[21][:-1])
                self.speed_mode = int(res[22][:-1])
                self.speed_interval = int(res[23][:-1])
                self.corr_map_name = res[24][:-1]
                self.data_out = int(res[25][:-1])
                self.data_name = res[26][:-1]
                self.corr_map_out = int(res[27][:-1])
                self.calib_left = res[28][:-1]
                self.calib_right = res[29][:-1]
                self.calib_target = res[30][:-1]
                self.calib_rows = int(res[31][:-1])
                self.calib_columns = int(res[32][:-1])
                self.calib_scale = float(res[33][:-1])
                self.data_xyz_name = res[34][:-1]
                self.ref_pcf = res[35][:-1]
                self.max_tmod = float(res[36][:-1])
                self.color_recon = int(res[37][:-1])
            except(ValueError, IndexError,Exception):
                print("Invalid values found in existing configuration file, rebuilding configuration file.")
                self.make_config()
        else:
            self.make_config()
