# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 17:49:34 2023

@author: myuey
"""
import os
class ConfigHandler():
    '''
    Stores inputs and settings in a text file for later retrieval.
    Deletion of the file causes the program to reset to stored values here in init.
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
        self.config_filename = "recon_config.txt"
        self.mask_thresh = 10 #15
        self.output = "recon.ply" #16
        self.f_file = "f.txt" #17
        self.speed_mode = 0 #18
        self.speed_interval = 10 #19
        self.corr_map_name = "correlation_map.png" #20
        self.data_out = 0#21
        self.data_name = "corr_data.txt" #22
        self.corr_map_out = 0#23
        self.calib_left = "cam1" #24
        self.calib_right = "cam2"  #25
        self.calib_target = "calib_mtx/" #26
        self.calib_rows = 8 #27
        self.calib_columns = 12 #28
        self.calib_scale = 0.004 #29
        self.color_recon = 0 #30
        self.f_mat_thresh = 0.7 #31
        self.sing_img_folder = "images/" #32
        self.sing_img_mode = 0 #33
        self.sing_left_ind = "cam1"#34
        self.sing_right_ind = "cam2"#35
        self.sing_ext = ".jpg"#36
        self.multi_recon = 0 #37
        self.f_search = 0 #38
        self.f_calc_mode = 0 #39
        self.f_mat_file_mode = 0 #40
        self.f_mat_ncc = 0 #41
        self.calib_img = 'calib_img/' #42
        self.interp_mode = 1 #43
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
        config_file.write(str(self.mask_thresh) + "\n")
        config_file.write(self.output + "\n")
        config_file.write(self.f_file + "\n")
        
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
        config_file.write(str(self.color_recon) + "\n")
        config_file.write(str(self.f_mat_thresh) + "\n")
        config_file.write(self.sing_img_folder + "\n")
        config_file.write(str(self.sing_img_mode) + "\n")
        config_file.write(self.sing_left_ind  + "\n")
        config_file.write(self.sing_right_ind  + "\n")
        config_file.write(self.sing_ext + "\n")
        config_file.write(str(self.multi_recon) + '\n')
        config_file.write(str(self.f_search) + '\n')
        config_file.write(str(self.f_calc_mode) + '\n')
        config_file.write(str(self.f_mat_file_mode)+ "\n")
        config_file.write(str(self.f_mat_ncc) + "\n")
        config_file.write(self.calib_img + "\n")
        config_file.write(str(self.interp_mode)+'\n')
        config_file.close()
        
    def load_config(self):
        '''
        If config file exists, read it and store values. 
        If reading causes errors, replace existing file with new file using default values to fill errors. 
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
                self.mask_thresh = int(res[15][:-1])
                self.output = res[16][:-1]
                self.f_file = res[17][:-1]
                self.speed_mode = int(res[18][:-1])
                self.speed_interval = int(res[19][:-1])
                self.corr_map_name = res[20][:-1]
                self.data_out = int(res[21][:-1])
                self.data_name = res[22][:-1]
                self.corr_map_out = int(res[23][:-1])
                self.calib_left = res[24][:-1]
                self.calib_right = res[25][:-1]
                self.calib_target = res[26][:-1]
                self.calib_rows = int(res[27][:-1])
                self.calib_columns = int(res[28][:-1])
                self.calib_scale = float(res[29][:-1])
                self.color_recon = int(res[30][:-1])
                self.f_mat_thresh = float(res[31][:-1])
                self.sing_img_folder = res[32][:-1]
                self.sing_img_mode = int(res[33][:-1])
                self.sing_left_ind = res[34][:-1]
                self.sing_right_ind = res[35][:-1]
                self.sing_ext = res[36][:-1]
                self.multi_recon = int(res[37][:-1])
                self.f_search = int(res[38][:-1])
                self.f_calc_mode= int(res[39][:-1])
                self.f_mat_file_mode = int(res[40][:-1])
                self.f_mat_ncc = int(res[41][:-1])
                self.calib_img = res[42][:-1]
                self.interp_mode = int(res[43][:-1]) 
            except(ValueError, IndexError,Exception):
                print("Invalid values found in existing configuration file, rebuilding configuration file.")
                self.make_config()
        else:
            self.make_config()
