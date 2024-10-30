# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:58:58 2024

@author: myuey
cap_window is a wrapper for turntable_gui.py code by Gregor_Gentsch
"""

import tkinter as tk
from tkinter import filedialog,ttk
import configinter as cinter
import ncc_core as ncc
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import scripts as scr
import matplotlib.pyplot as plt
from PIL import ImageTk, Image  
import numpy as np
import cv2
#from vimba import *
import time
import sys
from telemetrix import telemetrix
config = cinter.ConfigInter()
config.load_config()



version = 0.1

root = tk.Tk()
root.title("Stereo Reconstruction FSU Jena - v" + str(version))
rec_win_state = False
cal_win_state = False
cap_win_state = False
def toggle_rec_window():
    global rec_win_state

    if not rec_win_state:
        recon_window()
        rec_win_state = True
def recon_window():
    
    recon = tk.Toplevel(root)
    recon.title("3D Stereo Reconstruction -MG-")
    recon.geometry('705x370')
    recon.resizable(width=False, height=False)
    recon.focus_force()
    global set_win_state
    set_win_state = False
    def on_close():
        global rec_win_state
        rec_win_state = False

        recon.destroy()
    recon.protocol("WM_DELETE_WINDOW", on_close) 
    #Folder String Variables
    mat_fold = tk.StringVar(recon)
    imgL_fold = tk.StringVar(recon)
    imgR_fold = tk.StringVar(recon)
    sinFol_fold = tk.StringVar(recon)
    #Checkbox boolean variables
    sing_bool = tk.BooleanVar(recon)
    sing_bool.set(config.sing_img_mode)
    multi_bool = tk.BooleanVar(recon)
    multi_bool.set(config.multi_recon)
    f_mat_file_int = tk.IntVar(recon)
    f_mat_file_int.set(config.f_mat_file_mode)
    interp_mode_int = tk.IntVar(recon)
    interp_mode_int.set(config.interp_mode)
    speed_bool = tk.BooleanVar(recon)
    speed_bool.set(config.speed_mode)
    data_bool = tk.BooleanVar(recon)
    data_bool.set(config.data_out)
    rec_prev_bool = tk.BooleanVar(recon)
    rec_prev_bool.set(True)
    mask_prev_bool = tk.BooleanVar(recon)
    mask_prev_bool.set(True)
    map_out_bool = tk.BooleanVar(recon)
    map_out_bool.set(config.corr_map_out)
    recon_color_bool = tk.BooleanVar(recon)
    recon_color_bool.set(config.color_recon)
    f_search_bool = tk.BooleanVar(recon)
    f_search_bool.set(config.f_search)

    f_calc_mode = tk.IntVar(recon)
    f_calc_mode.set(config.f_calc_mode)

    f_ncc_bool = tk.BooleanVar(recon)
    f_ncc_bool.set(config.f_mat_ncc)

    dist_bool = tk.BooleanVar(recon)
    dist_bool.set(config.distort_comp)


    cuda_gpu_bool = tk.BooleanVar(recon)

    if(scr.get_gpu_name() == None):
        print('No CUDA GPU Detected.')
        cuda_gpu_bool.set(False)
    else:
        print('CUDA GPU Detected: ' + scr.get_gpu_name())
        cuda_gpu_bool.set(True)


    #output filebox
    out_lbl = tk.Label(recon, text = "Output File:")
    out_lbl.grid(sticky="E", row=0, column=0)
    out_txt = tk.Text(recon, height=1, width=35)
    out_txt.insert(tk.END, config.output)
    out_txt.grid(row=0, column=1)

    #matrix folder location
    mat_lbl = tk.Label(recon, text = "Matrices:")
    mat_lbl.grid(sticky="E",row = 1, column = 0)
    mat_txt = tk.Text(recon, height = 1, width = 35)
    mat_txt.insert(tk.END, config.mat_folder)
    mat_txt.grid(sticky="E", row = 1, column = 1)
    def mat_btn_click():
        folder_path = filedialog.askdirectory()
        mat_fold.set(folder_path + "/")
        mat_txt.delete('1.0', tk.END)
        mat_txt.insert('1.0', folder_path + "/")
        recon.focus_force()
    mat_btn = tk.Button(recon, text = "Browse", command = mat_btn_click)
    mat_btn.grid(sticky="W",row = 1, column = 2)

    #images_L location
    imgL_lbl = tk.Label(recon, text = "Left Images:")
    imgL_lbl.grid(sticky="E", row = 2, column = 0)
    imgL_txt = tk.Text(recon, height = 1, width = 35)
    imgL_txt.insert(tk.END, config.left_folder)
    imgL_txt.grid(row = 2, column = 1)
    def imgL_btn_click():
        folder_path = filedialog.askdirectory()
        imgL_fold.set(folder_path + "/")
        imgL_txt.delete('1.0', tk.END)
        imgL_txt.insert('1.0', folder_path + "/")
        recon.focus_force()
    imgL_btn = tk.Button(recon, text = "Browse", command = imgL_btn_click)
    imgL_btn.grid(sticky="W",row = 2, column = 2)

    #images_R location
    imgR_lbl = tk.Label(recon, text = "Right Images:")
    imgR_lbl.grid(sticky="E", row = 3, column = 0)
    imgR_txt = tk.Text(recon, height = 1, width = 35)
    imgR_txt.insert(tk.END, config.right_folder)
    imgR_txt.grid(row = 3, column = 1)
    def imgR_btn_click():
        folder_path = filedialog.askdirectory()
        imgR_fold.set(folder_path + "/")
        imgR_txt.delete('1.0', tk.END)
        imgR_txt.insert('1.0', folder_path + "/")
        recon.focus_force()
    imgR_btn = tk.Button(recon, text = "Browse", command = imgR_btn_click)
    imgR_btn.grid(sticky="W",row = 3, column = 2)

    #Single image folder location
    sinFol_lbl = tk.Label(recon, text = "Single Folder:")
    sinFol_lbl.grid(sticky="E", row = 4, column = 0)
    sinFol_txt = tk.Text(recon, height = 1, width = 35)
    sinFol_txt.insert(tk.END, config.sing_img_folder)
    sinFol_txt.grid(row = 4, column = 1)
    def sinFol_btn_click():
        folder_path = filedialog.askdirectory()
        sinFol_fold.set(folder_path + "/")
        sinFol_txt.delete('1.0', tk.END)
        sinFol_txt.insert('1.0', folder_path + "/")
        recon.focus_force()
    sinFol_btn = tk.Button(recon, text = "Browse", command = sinFol_btn_click)
    sinFol_btn.grid(sticky="W",row = 4, column = 2)
    #single image extension
    sinExt_lbl = tk.Label(recon, text = "Single Folder Extension:")
    sinExt_lbl.grid(sticky="E", row = 5, column = 0)
    sinExt_txt = tk.Text(recon, height = 1, width = 35)
    sinExt_txt.insert(tk.END, config.sing_ext)
    sinExt_txt.grid(row = 5, column = 1)

    #single image left indicator
    sinLeft_lbl = tk.Label(recon, text = "Single Folder Left Ind:")
    sinLeft_lbl.grid(sticky="E", row = 6, column = 0)
    sinLeft_txt = tk.Text(recon, height = 1, width = 35)
    sinLeft_txt.insert(tk.END, config.sing_left_ind)
    sinLeft_txt.grid(row = 6, column = 1)
    #single image right indicator
    sinRight_lbl = tk.Label(recon, text = "Single Folder Right Ind:")
    sinRight_lbl.grid(sticky="E", row = 7, column = 0)
    sinRight_txt = tk.Text(recon, height = 1, width = 35)
    sinRight_txt.insert(tk.END, config.sing_right_ind)
    sinRight_txt.grid(row = 7, column = 1)
    #single image mode checkbox
    sing_box= tk.Checkbutton(recon, text="Single Folder Mode", variable=sing_bool)
    sing_box.grid(sticky="W",row = 4, column = 3)
    #interpolation points input
    interp_lbl = tk.Label(recon, text = "Interpolations:")
    interp_lbl.grid(sticky="E", row = 8, column = 0)
    interp_txt = tk.Text(recon, height = 1, width = 35)
    interp_txt.insert(tk.END, config.interp)
    interp_txt.grid(row = 8, column = 1)

    #offset values input
    ofsXL_lbl = tk.Label(recon, text = "Offset X Left:")
    ofsXL_lbl.grid(sticky="E", row = 10, column = 0)
    ofsXL_txt = tk.Text(recon, height = 1, width = 35)
    ofsXL_txt.insert(tk.END, config.x_offset_L)
    ofsXL_txt.grid(row = 10, column = 1)

    ofsXR_lbl = tk.Label(recon, text = "Offset X Right:")
    ofsXR_lbl.grid(sticky="E", row = 11, column = 0)
    ofsXR_txt = tk.Text(recon, height = 1, width = 35)
    ofsXR_txt.insert(tk.END, config.x_offset_R)
    ofsXR_txt.grid(row = 11, column = 1)

    ofsYT_lbl = tk.Label(recon, text = "Offset Y Top:")
    ofsYT_lbl.grid(sticky="E", row = 12, column = 0)
    ofsYT_txt = tk.Text(recon, height = 1, width = 35)
    ofsYT_txt.insert(tk.END, config.y_offset_T)
    ofsYT_txt.grid(row = 12, column = 1)

    ofsYB_lbl = tk.Label(recon, text = "Offset Y Bottom:")
    ofsYB_lbl.grid(sticky="E", row = 13, column = 0)
    ofsYB_txt = tk.Text(recon, height = 1, width = 35)
    ofsYB_txt.insert(tk.END, config.y_offset_B)
    ofsYB_txt.grid(row = 13, column = 1)
    #F-mat finding threshold 
    fth_lbl = tk.Label(recon, text = "F Matrix Threshold:")
    fth_lbl.grid(sticky="E", row = 3, column = 5)
    fth_txt = tk.Text(recon, height = 1, width = 10)
    fth_txt.insert(tk.END, config.f_mat_thresh)
    fth_txt.grid(row = 4, column = 5)

    #correlation map 
    map_lbl = tk.Label(recon, text = "Correlation Map File:")
    map_lbl.grid(sticky="E", row = 9, column = 0)
    map_txt = tk.Text(recon, height = 1, width = 35)
    map_txt.insert(tk.END, config.corr_map_name)
    map_txt.grid(row = 9, column = 1)

    #check if folder contains folders
    def check_folder(path):
        contents = os.listdir(path)
        for item in contents:
            if not os.path.isdir(os.path.join(path, item)):
                return False
        return True
    def create_r_t():

        #Retrieve file names and data folder locations, need images and f mat if it is being loaded
        mat_folder = mat_txt.get('1.0', tk.END).rstrip()
        r_file_path = mat_folder + config.R_file
        t_file_path = mat_folder + config.t_file
        kL_file_path = mat_folder + config.kL_file
        kR_file_path = mat_folder + config.kR_file
        kL_rec = np.loadtxt(kL_file_path, skiprows=config.skiprow, delimiter = config.delim)
        kR_rec = np.loadtxt(kR_file_path, skiprows=config.skiprow, delimiter = config.delim)
        if(sing_bool.get()):
            img_folder_rec = sinFol_txt.get('1.0', tk.END).rstrip()
            imgL_ind_rec = sinLeft_txt.get('1.0', tk.END).rstrip()
            imgR_ind_rec = sinRight_txt.get('1.0', tk.END).rstrip()
            img_ext_rec = sinExt_txt.get('1.0', tk.END).rstrip()
            imgsL_rec,imgsR_rec = scr.load_images_1_dir(img_folder_rec, imgL_ind_rec, imgR_ind_rec, img_ext_rec)
        else:
            imgL_folder_rec = imgL_txt.get('1.0', tk.END).rstrip()
            imgR_folder_rec = imgR_txt.get('1.0', tk.END).rstrip()
            imgsL_rec, imgsR_rec = scr.load_images(imgL_folder_rec, imgR_folder_rec)
        #Find f mat and matching points using current settings
        Fmat_rec = None
        #Check if ncc mode is enabled
        if f_ncc_bool.get():
            Fmat_rec, pts1_rec,pts2_rec = scr.find_f_mat_ncc(imgsL_rec,imgsR_rec,thresh = config.f_mat_thresh, f_calc_mode = config.f_calc_mode, ret_pts = True) 
        else:
            if f_search_bool.get():
                Fmat_rec, pts1_rec,pts2_rec = scr.find_f_mat_list(imgsL_rec,imgsR_rec, thresh = config.f_mat_thresh, f_calc_mode = config.f_calc_mode, ret_pts = True)
            else:
                Fmat_rec, pts1_rec,pts2_rec = scr.find_f_mat(imgsL_rec[0],imgsR_rec[0], thresh = config.f_mat_thresh, f_calc_mode = config.f_calc_mode, ret_pts = True)
        #run correlation calibrate to find R and t
        R_rec, t_rec = scr.corr_calibrate(pts1_rec,pts2_rec,Fmat_rec, kL_rec, kR_rec)
        #save files
        np.savetxt(r_file_path, R_rec, header = "3\n3")
        np.savetxt(t_file_path, t_rec, header = "1\n3")
    #Error messages and handling for invalid entries on main screen
    def entry_check_main():
        error_flag = False
        verif_left = False
        verif_right = False
        fm_thr_chk = fth_txt.get('1.0', tk.END).rstrip()
        mat_fol_chk = mat_txt.get('1.0', tk.END).rstrip()
        
        

        if (mat_fol_chk[-1] != "/"):
            tk.messagebox.showerror("Invalid Input", "Matrix Folder must end in '/'")
            error_flag = True
        elif(not os.path.isdir(mat_fol_chk)):
            tk.messagebox.showerror("Folder Not Found", "Specified Matrix Folder '" + mat_fol_chk +
                                      "' not found.")
            error_flag = True
        #Check existence of matrix text files
        elif(not os.path.isfile(mat_fol_chk + config.kL_file)):#camera left
            tk.messagebox.showerror("File Not Found", "Specified Left Camera Matrix file '" +mat_fol_chk 
                                         + config.kL_file + "' not found.")
            error_flag = True
        elif(not os.path.isfile(mat_fol_chk + config.kR_file)):#camera right
            tk.messagebox.showerror("File Not Found", "Specified Right Camera Matrix file '" +mat_fol_chk 
                                         + config.kR_file + "' not found.")
            error_flag = True
            
        if(dist_bool.get()):
            #check for presence of distortion compensation vectors if needed
            if(not os.path.isfile(mat_fol_chk + config.left_distort)):
                tk.messagebox.showerror("File Not Found", "Specified left camera distortion file '" +mat_fol_chk 
                                             + config.left_distort + "' not found.")
                error_flag = True
            if(not os.path.isfile(mat_fol_chk + config.right_distort)):
                tk.messagebox.showerror("File Not Found", "Specified right camera distortion file '" +mat_fol_chk 
                                                 + config.right_distort + "' not found.")
                error_flag = True
                
        #If load fmat is true, check existence of specified f matrix file
        if f_mat_file_int.get() == 1:
            if(not os.path.isfile(mat_fol_chk + config.f_file)):
                tk.messagebox.showerror("File Not Found", "Specified Fundamental Matrix file '" +mat_fol_chk 
                                             + config.f_file + "' not found.")
                error_flag = True
        try:
            value = float(fm_thr_chk)
            if value >1 or value < 0:
                tk.messagebox.showerror("Invalid Input", "Fundamental Matrix Threshold must be float between 0 and 1")
                error_flag = True
                
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Fundamental Matrix Threshold must be float between 0 and 1")
            error_flag = True  
        
        if(sing_bool.get()):
            sin_fol_chk = sinFol_txt.get('1.0', tk.END).rstrip()
            if (sin_fol_chk[-1] != "/"):
                tk.messagebox.showerror("Invalid Input", "Single Image Folder must end in '/'")
                error_flag = True
            elif(not os.path.isdir(sin_fol_chk)):
                tk.messagebox.showerror("Folder Not Found", "Specified Image Folder '" + sin_fol_chk +
                                          "' not found.")
                error_flag = True
            elif(scr.check_balance_1_dir(sin_fol_chk, sinLeft_txt.get('1.0', tk.END).rstrip(), 
                                         sinRight_txt.get('1.0', tk.END).rstrip(),sinExt_txt.get('1.0', tk.END).rstrip())):
                tk.messagebox.showerror("Invalid Image Quantities", "Specified Folder, Extension, and Indicators result in invalid image quantities.")
                error_flag = True
        else:
            
            imgL_chk = imgL_txt.get('1.0', tk.END).rstrip()
            if (imgL_chk[-1] != "/"):
                tk.messagebox.showerror("Invalid Input", "Left Images Folder must end in  '/'")
                error_flag = True
            elif(not os.path.isdir(imgL_chk)):
                tk.messagebox.showerror("Folder Not Found", "Specified Left Images Folder '" + imgL_chk +
                                          "' not found.")
                error_flag = True
            elif not check_folder(imgL_chk) and multi_bool.get():
                tk.messagebox.showerror("Folder Error", "Multiple Runs mode selected but specified Left Images Folder '" + imgL_chk +
                                          "' does not contain only folders.")
                error_flag = True
            elif check_folder(imgL_chk) and not multi_bool.get():
                tk.messagebox.showerror("Folder Error", "Multiple Runs mode not selected but specified Left Images Folder '" + imgL_chk +
                                          "' contains only folders.")
                error_flag = True
            else:
                verif_left = True
                
            imgR_chk = imgR_txt.get('1.0', tk.END).rstrip()
            if (imgR_chk[-1] != "/"):
                tk.messagebox.showerror("Invalid Input", "Right Images Folder must end in  '/'")
                error_flag = True
            elif(not os.path.isdir(imgR_chk)):
                tk.messagebox.showerror("Folder Not Found", "Specified Right Images Folder '" + imgR_chk +
                                          "' not found.")
        
                error_flag = True
            elif not check_folder(imgR_chk) and multi_bool.get():
                tk.messagebox.showerror("Folder Error", "Multiple Runs mode selected but specified Right Images Folder '" + imgR_chk +
                                          "' does not contain only folders.")
                error_flag = True
            elif check_folder(imgR_chk) and not multi_bool.get():
                tk.messagebox.showerror("Folder Error", "Multiple Runs mode not selected but specified Right Images Folder '" + imgR_chk +
                                          "' contains only folders.")
                error_flag = True
            else:
                verif_right = True
        
            if(verif_left and verif_right):
                left_len = len(os.listdir(imgL_chk))
                right_len = len(os.listdir(imgR_chk))
                if left_len != right_len:
                    tk.messagebox.showerror("Mismatched Image Source", "Number of directories in '" + imgL_chk + "' and '" + 
                                            imgR_chk + "' do not match.")
                    error_flag = True
        if(not os.path.isfile(mat_fol_chk + config.R_file) and not os.path.isfile(mat_fol_chk + config.t_file) and not error_flag):
            print("Specified Rotation and Translation matrices not found. They will be calculated and saved to the filenames given.")  
            create_r_t()          
        elif(not os.path.isfile(mat_fol_chk + config.R_file)):#r mat
            tk.messagebox.showerror("File Not Found", "Specified Rotation Matrix file '" +mat_fol_chk 
                                         + config.R_file + "' not found.")
            error_flag = True
        elif(not os.path.isfile(mat_fol_chk + config.t_file)):#t vec
            tk.messagebox.showerror("File Not Found", "Specified Translation Vector file '" +mat_fol_chk 
                                         + config.t_file + "' not found.")
            error_flag = True
        interp_chk = interp_txt.get('1.0', tk.END).rstrip()
        try:
            value = int(interp_chk)
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Interpolations value must be integer")
            error_flag = True
        if(not error_flag):
            #Take dimensions of images and check against offsets
            img1 = None
            img2 = None
            if sing_bool.get():
                img1, img2 = scr.load_first_pair_1_dir(sinFol_txt.get('1.0', tk.END).rstrip(),sinLeft_txt.get('1.0', tk.END).rstrip(), 
                                                       sinRight_txt.get('1.0', tk.END).rstrip(), sinExt_txt.get('1.0', tk.END).rstrip())
                if img1.shape[0] == 0 or img2.shape[0] == 0:
                    tk.messagebox.showerror("Invalid Image Reference", "Specified Folder, Extension, and Indicators result in invalid images.")
                    error_flag = True
            else:
                
                img1, img2 = scr.load_first_pair(imgL_chk, imgR_chk)
            if(img1.shape == img2.shape):
                
                x_offL_chk = ofsXL_txt.get('1.0', tk.END).rstrip()
                try:
                    value = int(x_offL_chk)
                    if value <= 0 or value >= img1.shape[1]:
                        tk.messagebox.showerror("Invalid Input", "X L Offset value must be integer > 0 and < " + str(img1.shape[1]))
                        error_flag = True
                except ValueError:
                    tk.messagebox.showerror("Invalid Input", "X L Offset value must be integer > 0 and < " + str(img1.shape[1]))
                    error_flag = True
                x_offR_chk = ofsXR_txt.get('1.0', tk.END).rstrip()
                try:
                    value = int(x_offR_chk)
                    if value <= 0 or value >= img1.shape[1]:
                        tk.messagebox.showerror("Invalid Input", "X R Offset value must be integer > 0 and < " + str(img1.shape[1]))
                        error_flag = True
                except ValueError:
                    tk.messagebox.showerror("Invalid Input", "X R Offset value must be integer > 0 and < " + str(img1.shape[1]))
                    error_flag = True
                
                y_offT_chk = ofsYT_txt.get('1.0', tk.END).rstrip()
                try:
                    value = int(y_offT_chk)
                    if value <= 0 or value >= img1.shape[0]:
                        tk.messagebox.showerror("Invalid Input", "Y T Offset value must be integer > 0 and < " + str(img1.shape[0]))
                        error_flag = True
                except ValueError:
                    tk.messagebox.showerror("Invalid Input", "Y T Offset value must be integer > 0 and < " + str(img1.shape[0]))
                    error_flag = True
                y_offB_chk = ofsYB_txt.get('1.0', tk.END).rstrip()
                try:
                    value = int(y_offB_chk)
                    if value <= 0 or value >= img1.shape[0]:
                        tk.messagebox.showerror("Invalid Input", "Y B Offset value must be integer > 0 and < " + str(img1.shape[0]))
                        error_flag = True
                except ValueError:
                    tk.messagebox.showerror("Invalid Input", "Y B Offset value must be integer > 0 and < " + str(img1.shape[0]))
                    error_flag = True 
            else:
                tk.messagebox.showerror("Image Error", "Images are not the same shape")
                error_flag = True
                
        return error_flag

    #Previews the first pair of images in the specified directories
    #Used for checking if the fundamental matrix rectifies the images correctly
    def preview_window():
        global prev_disp
        entry_chk = entry_check_main()
        
        
        
        if not entry_chk:
            if prev_disp and prev_disp.winfo_exists(): #Creates new updated window and destroys old one 
                plt.close()
                prev_disp.destroy()
            prev_disp = tk.Toplevel(recon)
            prev_disp.title("Preview")
            prev_disp.geometry('1000x500')
            prev_disp.focus_force()
            def on_close():
                plt.close()
                prev_disp.destroy()
            prev_disp.protocol("WM_DELETE_WINDOW", on_close) 
            prev_disp.resizable(width=False, height=False)
            config.mat_folder = mat_txt.get('1.0', tk.END).rstrip()
            if sing_bool.get():
                config.sing_img_folder = sinFol_txt.get('1.0', tk.END).rstrip()
                config.sing_left_ind = sinLeft_txt.get('1.0', tk.END).rstrip()
                config.sing_right_ind = sinRight_txt.get('1.0', tk.END).rstrip()
                config.sing_ext = sinExt_txt.get('1.0', tk.END).rstrip()
            else:
                config.left_folder = imgL_txt.get('1.0', tk.END).rstrip()
                config.right_folder = imgR_txt.get('1.0', tk.END).rstrip()
            
            config.x_offset_L = int(ofsXL_txt.get('1.0', tk.END).rstrip())
            config.x_offset_R = int(ofsXR_txt.get('1.0', tk.END).rstrip())
            config.y_offset_T = int(ofsYT_txt.get('1.0', tk.END).rstrip())
            config.y_offset_B = int(ofsYB_txt.get('1.0', tk.END).rstrip())
            config.f_mat_thresh = float(fth_txt.get('1.0', tk.END).rstrip())
            if rec_prev_bool.get():
                
                fund_mat = None
                imPL = None
                imPR = None
                if sing_bool.get():
                    imPL,imPR = scr.load_first_pair_1_dir(config.sing_img_folder,config.sing_left_ind, config.sing_right_ind, config.sing_ext)
                else:
                    imPL,imPR = scr.load_first_pair(config.left_folder,config.right_folder)
                if os.path.isfile(config.mat_folder + config.f_file) and config.f_mat_file_mode == 1:
                    fund_mat = np.loadtxt(config.mat_folder + config.f_file, skiprows=config.skiprow, delimiter = config.delim)
                    print("Fundamental Matrix Loaded From File: " + config.mat_folder + config.f_file)
                else:
                    if f_search_bool.get() and not f_ncc_bool.get():
                        imgL = None
                        imgR = None
                        if(sing_bool.get()):
                            imgL,imgR = scr.load_images_1_dir(config.sing_img_folder, config.sing_left_ind, config.sing_right_ind, config.sing_ext)
                        else:
                            imgL,imgR = scr.load_images(folderL = config.left_folder, folderR = config.right_folder)
                        fund_mat = scr.find_f_mat_list(imgL,imgR, thresh = float(fth_txt.get('1.0', tk.END).rstrip()), f_calc_mode = f_calc_mode.get())
                    else:
                        if sing_bool.get():
                            imL,imR = scr.load_first_pair_1_dir(config.sing_img_folder,config.sing_left_ind, config.sing_right_ind, config.sing_ext)
                        else:
                            imL,imR = scr.load_first_pair(config.left_folder,config.right_folder)
                        if f_ncc_bool.get():
                            if(sing_bool.get()):
                                imgL,imgR = scr.load_images_1_dir(config.sing_img_folder, config.sing_left_ind, config.sing_right_ind, config.sing_ext)
                            else:
                                imgL,imgR = scr.load_images(folderL = config.left_folder, folderR = config.right_folder)
                            fund_mat = scr.find_f_mat_ncc(imgL,imgR,thresh = float(fth_txt.get('1.0', tk.END).rstrip()), f_calc_mode = f_calc_mode.get())
                        else:
                            fund_mat = scr.find_f_mat(imL,imR, thresh = float(fth_txt.get('1.0', tk.END).rstrip()), f_calc_mode = f_calc_mode.get())
                
                try:
                    im1,im2, H1, H2 = scr.rectify_pair(imPL,imPR, fund_mat)
                except(Exception):
                    print('Rectification failure. Check settings.')
                
                
            else:
                if sing_bool.get():
                    im1,im2 = scr.load_first_pair_1_dir(config.sing_img_folder,config.sing_left_ind, config.sing_right_ind, config.sing_ext)
                else:
                    im1,im2 = scr.load_first_pair(config.left_folder,config.right_folder)
            try:          
                if mask_prev_bool.get():
                    im1 = scr.mask_img(im1,config.mask_thresh)
                    im2 = scr.mask_img(im2,config.mask_thresh)
                    
                fig = scr.create_stereo_offset_fig(im1, im2, config.x_offset_L, config.x_offset_R, config.y_offset_T, config.y_offset_B)
                canvas = FigureCanvasTkAgg(fig, master = prev_disp)  
                canvas.draw()
                toolbar = NavigationToolbar2Tk(canvas, prev_disp, pack_toolbar=False)
                toolbar.update()
                toolbar.pack(side=tk.BOTTOM, fill=tk.X)
                canvas.get_tk_widget().pack()
            except(Exception):
                print('Preview Error. Check settings.')
    global prev_disp            
    prev_disp = None
    prev_btn = tk.Button(recon, text = "Preview", command = preview_window)
    prev_btn.grid(row = 0, column =5)



    #rectified preview checkbox
    rect_box = tk.Checkbutton(recon, text="Rectify Preview", variable=rec_prev_bool)
    rect_box.grid(sticky="W",row = 1, column = 5)
    #masked preview checkbox 
    mask_box = tk.Checkbutton(recon, text="Mask Preview", variable=mask_prev_bool)
    mask_box.grid(sticky="W",row =2, column = 5)
    #speed checkbox
    speed_box= tk.Checkbutton(recon, text="Increase Speed", variable=speed_bool)
    speed_box.grid(sticky="W",row = 5, column = 3)
    #corr map with recon checkbox
    cor_box= tk.Checkbutton(recon, text="Build Map", variable=map_out_bool)
    cor_box.grid(sticky="W",row =6, column = 3)
    #Full data checkbox
    data_box= tk.Checkbutton(recon, text="Data Out", variable=data_bool)
    data_box.grid(sticky="W",row =7, column = 3)
    #multi-recon checkbox
    multi_box = tk.Checkbutton(recon, text="Multiple Runs", variable=multi_bool)
    multi_box.grid(sticky="W",row = 8, column = 3)
    #color recon checkbox
    color_box = tk.Checkbutton(recon, text="Color Recon", variable=recon_color_bool)
    color_box.grid(sticky="W",row = 9, column = 3)

    #distortion compensation checkbox
    dist_box = tk.Checkbutton(recon, text = "Dist Comp", variable = dist_bool)
    dist_box.grid(sticky="W",row = 10, column = 3)

    #f mat search through all image pairs checkbox
    f_search_box = tk.Checkbutton(recon, text = "F Mat Verify", variable=f_search_bool)
    f_search_box.grid(sticky="W",row =5, column = 5)
    #f mat via ncc checkbox
    f_ncc_box = tk.Checkbutton(recon, text = "F Mat NCC", variable = f_ncc_bool)
    f_ncc_box.grid(sticky="W",row = 6, column = 5)



    tk.Radiobutton(recon, text="LMEDS", variable = f_calc_mode, value = 0).grid(sticky="W",row = 7, column = 5)
    tk.Radiobutton(recon, text="8POINT",  variable = f_calc_mode, value = 1).grid(sticky="W",row = 8, column = 5)
    tk.Radiobutton(recon, text="RANSAC", variable = f_calc_mode, value = 2).grid(sticky="W",row = 9, column = 5)


    #start button for main reconstruction
    def st_btn_click(): 
        entry_chk = entry_check_main()
        if not entry_chk and not multi_bool.get():
            print("Creating Reconstruction")
            config.mat_folder = mat_txt.get('1.0', tk.END).rstrip()
            config.sing_img_mode = int(sing_bool.get())
            if sing_bool.get():
                config.sing_img_folder = sinFol_txt.get('1.0', tk.END).rstrip()
                config.sing_left_ind = sinLeft_txt.get('1.0', tk.END).rstrip()
                config.sing_right_ind = sinRight_txt.get('1.0', tk.END).rstrip()
                config.sing_ext = sinExt_txt.get('1.0', tk.END).rstrip()
            else:
                config.left_folder = imgL_txt.get('1.0', tk.END).rstrip()
                config.right_folder = imgR_txt.get('1.0', tk.END).rstrip()
            config.interp = int(interp_txt.get('1.0', tk.END).rstrip())
            config.x_offset_L = int(ofsXL_txt.get('1.0', tk.END).rstrip())
            config.x_offset_R = int(ofsXR_txt.get('1.0', tk.END).rstrip())
            config.y_offset_T = int(ofsYT_txt.get('1.0', tk.END).rstrip())
            config.y_offset_B = int(ofsYB_txt.get('1.0', tk.END).rstrip())
            config.f_mat_thresh = float(fth_txt.get('1.0', tk.END).rstrip())
            config.output = out_txt.get('1.0', tk.END).rstrip()
            config.speed_mode = speed_bool.get()
            config.data_out = data_bool.get()
            config.corr_map_out = map_out_bool.get()
            config.corr_map_name = map_txt.get('1.0', tk.END).rstrip()
            ncc.run_cor(config)
        elif not entry_chk and multi_bool.get():
            
            print("Creating Multiple Reconstructions")
            config.f_mat_thresh = float(fth_txt.get('1.0', tk.END).rstrip())
            config.mat_folder = mat_txt.get('1.0', tk.END).rstrip()
            config.speed_mode = speed_bool.get()
            config.interp = int(interp_txt.get('1.0', tk.END).rstrip())
            config.x_offset_L = int(ofsXL_txt.get('1.0', tk.END).rstrip())
            config.x_offset_R = int(ofsXR_txt.get('1.0', tk.END).rstrip())
            config.y_offset_T = int(ofsYT_txt.get('1.0', tk.END).rstrip())
            config.y_offset_B = int(ofsYB_txt.get('1.0', tk.END).rstrip())
            out_base = out_txt.get('1.0', tk.END).rstrip()
            config.corr_map_name = map_txt.get('1.0', tk.END).rstrip()
            config.data_out = data_bool.get()
            config.corr_map_out = map_out_bool.get()
            if "." in out_base:
                out_base = out_base.split(".", 1)[0]
            config.sing_img_mode = int(sing_bool.get())    
            if(sing_bool.get()):
                sing_base = sinFol_txt.get('1.0', tk.END).rstrip()
                config.sing_left_ind = sinLeft_txt.get('1.0', tk.END).rstrip()
                config.sing_right_ind = sinRight_txt.get('1.0', tk.END).rstrip()
                config.sing_ext = sinExt_txt.get('1.0', tk.END).rstrip()
                counter = 0
                for a in sorted(os.listdir(sing_base)):
                    config.sing_img_folder = sing_base + a + "/"
                    if counter < 10000:
                        config.output = out_base + "{:04d}".format(counter)
                    else:
                        config.output = out_base + str(counter)
                    ncc.run_cor(config)
                    counter+=1
            else:
                pass
            
                left_base = imgL_txt.get('1.0', tk.END).rstrip()
                right_base = imgR_txt.get('1.0', tk.END).rstrip()
            
            
                counter = 0
            
                for a,b in zip(sorted(os.listdir(left_base)), sorted(os.listdir(right_base))):
                    config.left_folder = left_base + a + "/"
                    config.right_folder = right_base + b + "/"
                    left_len = len(os.listdir(config.left_folder))
                    right_len = len(os.listdir(config.right_folder))
                    if counter < 10000:
                        config.output = out_base + "{:04d}".format(counter)
                    else:
                        config.output = out_base + str(counter)
                    if left_len != right_len:
                        print("Reconstruction Error for Folders: '" + config.left_folder + "' and '" +
                          config.right_folder + "'. Mismatched image counts. This pair has been skipped.")
                    else:
                        ncc.run_cor(config)
                    counter+=1
                    

     
    st_btn = tk.Button(recon, text = "Start Reconstruction", command = st_btn_click)
    st_btn.grid(row = 14, column = 1)
    #stop_btn = tk.Button(recon, text = "Cancel Reconstruction", command = stop)
    #stop_btn.grid(row = 14, column = 0)
    #correlation map creation
    def cor_map_btn_click():
        
        entry_chk = entry_check_main()
        if not entry_chk:
            print("Creating Correlation Map")
            config.mat_folder = mat_txt.get('1.0', tk.END).rstrip()
            if sing_bool.get():
                config.sing_img_mode = 1
                config.sing_img_folder = sinFol_txt.get('1.0', tk.END).rstrip()
                config.sing_left_ind = sinLeft_txt.get('1.0', tk.END).rstrip()
                config.sing_right_ind = sinRight_txt.get('1.0', tk.END).rstrip()
                config.sing_ext = sinExt_txt.get('1.0', tk.END).rstrip()
            else:
                config.sing_img_mode = 0
                config.left_folder = imgL_txt.get('1.0', tk.END).rstrip()
                config.right_folder = imgR_txt.get('1.0', tk.END).rstrip()
            config.interp = int(interp_txt.get('1.0', tk.END).rstrip())
            config.x_offset_L = int(ofsXL_txt.get('1.0', tk.END).rstrip())
            config.x_offset_R = int(ofsXR_txt.get('1.0', tk.END).rstrip())
            config.y_offset_T = int(ofsYT_txt.get('1.0', tk.END).rstrip())
            config.y_offset_B = int(ofsYB_txt.get('1.0', tk.END).rstrip())
            config.corr_map_name = map_txt.get('1.0', tk.END).rstrip()
            config.speed_mode = speed_bool.get()
            ncc.run_cor(config, mapgen = True)
    map_btn = tk.Button(recon, text = "Create", command = cor_map_btn_click)
    map_btn.grid(row = 9, column = 2)
    #reset button
    def rst_btn_click():
        global config
        config = cinter.ConfigHandler()
        out_txt.delete('1.0', tk.END)
        out_txt.insert(tk.END, config.output)
        mat_txt.delete('1.0', tk.END)
        mat_txt.insert(tk.END, config.mat_folder)
        imgL_txt.delete('1.0', tk.END)
        imgL_txt.insert(tk.END, config.left_folder)
        imgR_txt.delete('1.0', tk.END)
        imgR_txt.insert(tk.END, config.right_folder)
        interp_txt.delete('1.0', tk.END)
        interp_txt.insert(tk.END, config.interp)
        ofsXL_txt.delete('1.0', tk.END)
        ofsXL_txt.insert(tk.END, config.x_offset_L)
        ofsXR_txt.delete('1.0', tk.END)
        ofsXR_txt.insert(tk.END, config.x_offset_R)
        ofsYT_txt.delete('1.0', tk.END)
        ofsYT_txt.insert(tk.END, config.y_offset_T)
        ofsYB_txt.delete('1.0', tk.END)
        ofsYB_txt.insert(tk.END, config.y_offset_B)
        map_txt.delete('1.0', tk.END)
        map_txt.insert(tk.END, config.corr_map_name)
        sinExt_txt.delete('1.0', tk.END)
        sinExt_txt.insert(tk.END, config.sing_ext)
        sinFol_txt.delete('1.0', tk.END)
        sinFol_txt.insert(tk.END, config.sing_img_folder)
        sinLeft_txt.delete('1.0', tk.END)
        sinLeft_txt.insert(tk.END, config.sing_left_ind)
        sinRight_txt.delete('1.0', tk.END)
        sinRight_txt.insert(tk.END, config.sing_right_ind)
        fth_txt.delete('1.0', tk.END)
        fth_txt.insert(tk.END, config.f_mat_thresh)
        sing_bool.set(config.sing_img_mode)
        speed_bool.set(config.speed_mode)
        map_out_bool.set(config.corr_map_out)
        data_bool.set(config.data_out)
        multi_bool.set(config.multi_recon)
        
    rst_btn = tk.Button(recon, text = "Reset", command = rst_btn_click)
    rst_btn.grid(row = 2, column = 3, sticky='e')
    #save all fields as default button
    def cfg_btn_click(): 
        config.output = out_txt.get('1.0',tk.END).rstrip()
        config.mat_folder = mat_txt.get('1.0', tk.END).rstrip()
        config.left_folder = imgL_txt.get('1.0', tk.END).rstrip()
        config.right_folder = imgR_txt.get('1.0', tk.END).rstrip()
        config.f_mat_thresh = fth_txt.get('1.0', tk.END).rstrip()
        config.interp = int(interp_txt.get('1.0', tk.END).rstrip())
        config.sing_img_folder = sinFol_txt.get('1.0', tk.END).rstrip()
        config.x_offset_L = int(ofsXL_txt.get('1.0', tk.END).rstrip())
        config.x_offset_R = int(ofsXR_txt.get('1.0', tk.END).rstrip())
        config.y_offset_T = int(ofsYT_txt.get('1.0', tk.END).rstrip())
        config.y_offset_B = int(ofsYB_txt.get('1.0', tk.END).rstrip())
        config.corr_map_name = map_txt.get('1.0',tk.END).rstrip()
        config.corr_map_out = int(map_out_bool.get()) 
        config.sing_img_mode = int(sing_bool.get()) 
        config.sing_ext = sinExt_txt.get('1.0',tk.END).rstrip()
        config.sing_left_ind = sinLeft_txt.get('1.0',tk.END).rstrip()
        config.sing_right_ind = sinRight_txt.get('1.0',tk.END).rstrip()
        config.speed_mode = int(speed_bool.get())
        config.multi_recon = int(multi_bool.get())
        config.data_out = int(data_bool.get())
        config.f_search = int(f_search_bool.get())
        config.f_calc_mode = int(f_calc_mode.get())
        config.make_config()   
        
    cfg_btn = tk.Button(recon, text = "Set Defaults", command = cfg_btn_click)
    cfg_btn.grid(row = 1, column = 3, sticky='e')

    #settings window
    
    def toggle_set_window():
        global set_win_state
        if not set_win_state:
            set_window()
            set_win_state = True

    def set_window():
        set_disp = tk.Toplevel(recon)
        set_disp.title("Settings")
        set_disp.geometry('380x340')
        set_disp.focus_force()
        set_disp.resizable(width=False, height=False)
        def on_close():
            global set_win_state
            set_win_state = False
            set_disp.destroy()
        set_disp.protocol("WM_DELETE_WINDOW", on_close) 
            
        tvec_lbl = tk.Label(set_disp, text = "t-Vector File:")
        tvec_txt = tk.Text(set_disp, height = 1, width = 20)
        tvec_txt.insert(tk.END, config.t_file)
        tvec_lbl.grid(sticky="E",row = 1, column = 0)
        tvec_txt.grid(row = 1, column = 1)
        
        Rmat_lbl = tk.Label(set_disp, text = "R-Matrix File:")
        Rmat_txt = tk.Text(set_disp, height = 1, width = 20)
        Rmat_txt.insert(tk.END, config.R_file)
        Rmat_lbl.grid(sticky="E",row = 2, column = 0)
        Rmat_txt.grid(row = 2, column = 1)
        
        lkp_lbl = tk.Label(set_disp, text = "Lineskips:")
        lkp_txt = tk.Text(set_disp, height = 1, width = 20)
        lkp_txt.insert(tk.END, config.skiprow)
        lkp_lbl.grid(sticky="E",row = 3, column = 0)
        lkp_txt.grid(row = 3, column = 1)
        
        kl_lbl = tk.Label(set_disp, text = "Left Camera Matrix File:")
        kl_txt = tk.Text(set_disp, height = 1, width = 20)
        kl_txt.insert(tk.END, config.kL_file)
        kl_lbl.grid(sticky="E",row = 4, column = 0)
        kl_txt.grid(row = 4, column = 1)
        
        kr_lbl = tk.Label(set_disp, text = "Right Camera Matrix File:")
        kr_txt = tk.Text(set_disp, height = 1, width = 20)
        kr_txt.insert(tk.END, config.kR_file)
        kr_lbl.grid(sticky="E",row = 5, column = 0)
        kr_txt.grid(row = 5, column = 1)
        
        f_lbl = tk.Label(set_disp, text = "Fundamental Matrix File:")
        f_txt = tk.Text(set_disp, height = 1, width = 20)
        f_txt.insert(tk.END, config.f_file)
        f_lbl.grid(sticky="E",row = 6, column = 0)
        f_txt.grid(row = 6, column = 1)
        
        delim_lbl = tk.Label(set_disp, text = "Delimiter:")
        delim_txt = tk.Text(set_disp, height = 1, width = 20)
        delim_txt.insert(tk.END, config.delim)
        delim_lbl.grid(sticky="E",row = 7, column = 0)
        delim_txt.grid(row = 7, column = 1)
        
        thr_lbl = tk.Label(set_disp, text = "Correlation Threshold:")
        thr_txt = tk.Text(set_disp, height = 1, width = 20)
        thr_txt.insert(tk.END, config.thresh)
        thr_lbl.grid(sticky="E",row = 8, column = 0)
        thr_txt.grid(row = 8, column = 1)
        
        msk_lbl = tk.Label(set_disp, text = "Mask Threshold:")
        msk_txt = tk.Text(set_disp, height = 1, width = 20)
        msk_txt.insert(tk.END, config.mask_thresh)
        msk_lbl.grid(sticky="E",row = 9, column = 0)
        msk_txt.grid(row = 9, column = 1)
        
        spd_lbl = tk.Label(set_disp, text = "Speed Interval:")
        spd_txt = tk.Text(set_disp, height = 1, width = 20)
        spd_txt.insert(tk.END, config.speed_interval)
        spd_lbl.grid(sticky="E",row = 10, column = 0)
        spd_txt.grid(row = 10, column = 1)
        
        dot_lbl = tk.Label(set_disp, text = "Data Out File:")
        dot_txt = tk.Text(set_disp, height = 1, width = 20)
        dot_txt.insert(tk.END, config.data_name)
        dot_lbl.grid(sticky="E",row = 11, column = 0)
        dot_txt.grid(row = 11, column = 1)
        
        distL_lbl = tk.Label(set_disp, text = "Left Distortion:")
        distL_txt = tk.Text(set_disp, height = 1, width = 20)
        distL_txt.insert(tk.END, config.left_distort)
        distL_lbl.grid(sticky="E",row = 12, column = 0)
        distL_txt.grid(row = 12, column = 1)
        
        distR_lbl = tk.Label(set_disp, text = "Right Distortion:")
        distR_txt = tk.Text(set_disp, height = 1, width = 20)
        distR_txt.insert(tk.END, config.right_distort)
        distR_lbl.grid(sticky="E",row = 13, column = 0)
        distR_txt.grid(row = 13, column = 1)
        
        
        inter_mode_lbl  = tk.Label(set_disp, text = "Interpolation Mode:")
        inter_mode_lbl.grid(sticky="E",row = 14, column = 0)
        tk.Radiobutton(set_disp, text="Radial Basis Function", variable = interp_mode_int, value = 1).grid(row = 14, column = 1)
        tk.Radiobutton(set_disp, text="Linear", variable = interp_mode_int, value = 0).grid(row = 14, column = 2)
        
        
        
        tk.Radiobutton(set_disp, text="Calc F", variable = f_mat_file_int, value = 0).grid(row = 6, column = 2)
        tk.Radiobutton(set_disp, text="Load F",  variable = f_mat_file_int, value = 1).grid(row = 7, column = 2)
        tk.Radiobutton(set_disp, text="Save F", variable = f_mat_file_int, value = 2).grid(row = 8, column = 2)

        

        def entry_check_settings():
            error_flag = False
            mat_fold = mat_txt.get('1.0', tk.END).rstrip()
            tvec_chk = tvec_txt.get('1.0',tk.END).rstrip()
            if dist_bool.get():
                distL_chk = distL_txt.get('1.0',tk.END).rstrip()
                distR_chk = distR_txt.get('1.0',tk.END).rstrip()
                if (not distL_chk.endswith(".txt")):
                    tk.messagebox.showerror("Invalid Input", "t-vector file type must be .txt.")
                    error_flag = True
                if (not distR_chk.endswith(".txt")):
                    tk.messagebox.showerror("Invalid Input", "t-vector file type must be .txt.")
                    error_flag = True
            if (not tvec_chk.endswith(".txt")):
                tk.messagebox.showerror("Invalid Input", "t-vector file type must be .txt.")
                error_flag = True
            Rmat_chk = Rmat_txt.get('1.0',tk.END).rstrip()
            if (not Rmat_chk.endswith(".txt")):
                tk.messagebox.showerror("Invalid Input", "R-matrix file type must be .txt.")
                error_flag = True
                
            f_chk = f_txt.get('1.0',tk.END).rstrip()
            if (not f_chk.endswith(".txt")):
                tk.messagebox.showerror("Invalid Input", "Fundamental matrix file type must be .txt.")
                error_flag = True
            skiprow_chk = lkp_txt.get('1.0',tk.END).rstrip()
            try:
                value = int(skiprow_chk)
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Lineskips value must be an integer.")
                error_flag = True
            kL_file_chk = kl_txt.get('1.0',tk.END).rstrip()
            if (not kL_file_chk.endswith(".txt")):
                tk.messagebox.showerror("Invalid Input", "Left Camera Matrix file type must be .txt.")
                error_flag = True
            kR_file_chk = kr_txt.get('1.0',tk.END).rstrip()
            if (not kR_file_chk.endswith(".txt")):
                tk.messagebox.showerror("Invalid Input", "Right Camera Matrix file type must be .txt.")
                error_flag = True
            dot_chk = dot_txt.get('1.0',tk.END).rstrip()
            if (not dot_chk.endswith(".txt")):
                tk.messagebox.showerror("Invalid Input", "Data out file type must be .txt.")
                error_flag = True
            thresh_chk = thr_txt.get('1.0',tk.END).rstrip()
            try:
                value = float(thresh_chk)
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Correlation Threshold value must be a float.")
                error_flag = True
            msk_chk = msk_txt.get('1.0',tk.END).rstrip()
            try:
                value = int(msk_chk)
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Mask Threshold value must be an integer.")
                error_flag = True
            spd_chk = spd_txt.get('1.0',tk.END).rstrip()   
            try:
                value = int(spd_chk)
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Speed Interval value must be an integer.")
                error_flag = True
                
            
            if error_flag:
                set_disp.focus_force()
            return error_flag
        def cnc_btn_click():
            global set_win_state
            set_win_state = False
            set_disp.destroy()
        cnc_btn = tk.Button(set_disp, text = "Cancel", command = cnc_btn_click)
        def ok_btn_click():
            if not entry_check_settings():
                config.t_file = tvec_txt.get('1.0',tk.END).rstrip()
                config.R_file = Rmat_txt.get('1.0',tk.END).rstrip()
                config.skiprow = int(lkp_txt.get('1.0',tk.END).rstrip())
                config.kL_file = kl_txt.get('1.0',tk.END).rstrip()
                config.kR_file = kr_txt.get('1.0',tk.END).rstrip()
                config.f_file = f_txt.get('1.0',tk.END).rstrip()
                if config.distort_comp:
                    config.left_distort = distL_txt.get('1.0',tk.END).rstrip()
                    config.right_distort = distR_txt.get('1.0',tk.END).rstrip()
                if(delim_txt.get('1.0',tk.END).rstrip() == ""):
                    config.delim = " "
                else:
                    config.delim = delim_txt.get('1.0',tk.END).rstrip()
                config.thresh = float(thr_txt.get('1.0',tk.END).rstrip())
                config.mask_thresh = int(msk_txt.get('1.0',tk.END).rstrip())
                config.f_mat_file_mode= f_mat_file_int.get()
                config.interp_mode = interp_mode_int.get()
                config.color_recon = int(recon_color_bool.get())
                config.speed_interval = int(spd_txt.get('1.0',tk.END).rstrip())
                config.data_name = dot_txt.get('1.0',tk.END).rstrip()
                global set_win_state
                set_win_state = False
                set_disp.destroy()
        ok_btn = tk.Button(set_disp, text = "OK", command = ok_btn_click)
        
        cnc_btn.grid(row = 15, column = 0)
        ok_btn.grid(row = 15,column = 1)

    set_btn = tk.Button(recon, text = "Settings", command = toggle_set_window)
    set_btn.grid(row = 3, column = 3, sticky='e')

def toggle_cap_window():
    global cap_win_state
    if not cap_win_state:
         cap_window()
         cap_win_state = True   
    
def cap_window():
    #establish board and pins
    dir_pin = config.dir_pin
    pul_pin = config.pul_pin
    '''
    # # Create a Telemetrix instance.
    board = telemetrix.Telemetrix()
    
    # Set the DIGITAL_PIN as an output pin
    board.set_pin_mode_digital_output(dir_pin)
    board.set_pin_mode_digital_output(pul_pin)
    '''
    
    #load in images
    '''
    pattern_path = "C:\\Users\\IAOB\Documents\\tinkering\\BBM_Adrian\\1000p2p20\\"
    pattern_path = config.pattern_path
    pattern_file_names = os.listdir(pattern_path)
    #comment next line out if using all patterns
    pattern_file_names = pattern_file_names[:100]
    pattern_file_list = []
    for pattern_name in pattern_file_names:
        pattern = Image.open(pattern_path+pattern_name)
        pattern_file_list.append(pattern)
        '''
    # capture images and save them in the desired folder
    def cam_capture(target_folder, position, pattern_num):
        expo_time = int(expo_time_entry.get())
        timeout_time = 100
        #timeout_time = int((2 * (expo_time/1000)) + (expo_time/5000))

        with Vimba.get_instance() as vimba:
            cams = vimba.get_all_cameras()
            with cams[0] as cam1:
                cam1.ExposureTime.set(expo_time)
                frame1 = cam1.get_frame(timeout_ms=timeout_time)
                frame1.convert_pixel_format(PixelFormat.Bgr8)
                cv2.imwrite(target_folder + "/" +  "cam1_pos_" + f"{position:04d}" + "pattern_" + f"{pattern_num:04d}" + ".jpg", 
                            frame1.as_opencv_image())

            with cams[1] as cam2:
                cam2.ExposureTime.set(expo_time)
                frame2 = cam2.get_frame(timeout_ms=timeout_time)
                frame2.convert_pixel_format(PixelFormat.Bgr8)
                cv2.imwrite(target_folder + "/" +  "cam2_pos_" + f"{position:04d}" + "pattern_" + f"{pattern_num:04d}" + ".jpg",
                            frame2.as_opencv_image())
        return

    '''
    # turns the turntable
    def turntable(steps_per_angle):
        for steps in range(steps_per_angle):
            board.digital_write(pul_pin, 1)
            time.sleep(0.005)
            board.digital_write(pul_pin, 0)
            time.sleep(0.005)
        return
    '''
    cap_win = tk.Toplevel(root)
    cap_win.title("Image Capture")
    cap_win.geometry("600x400")


    #Entry Window and Label for exposure time
    expo_time_value = tk.IntVar(value=config.expo_time)
    expo_time_label = ttk.Label(cap_win, text="Exposure time in µs", padding=[10])
    expo_time_label.grid(column=0, row=1)
    expo_time_entry = ttk.Entry(cap_win, text=expo_time_value)
    expo_time_entry.grid(column=1, row=1)


    #Entry Window and label for position start
    pos_start_value = tk.IntVar(value=config.start_pos)
    pos_start_label = ttk.Label(cap_win, text="Starting Position", padding=[10])
    pos_start_label.grid(column=2, row=1)
    pos_start_entry = ttk.Entry(cap_win, text=pos_start_value)
    pos_start_entry.grid(column=3, row=1)

    #Stepper motor Steps, Microstepping and such
    steps_per_rev = tk.IntVar(value=config.steps_rev)
    steps_per_rev_label = ttk.Label(cap_win, text="Steps per revolution small motor", padding=[10])
    steps_per_rev_label.grid(column=0, row=2)
    steps_per_rev_entry = ttk.Entry(cap_win, text=steps_per_rev)
    steps_per_rev_entry.grid(column=1, row=2)


    #gear ratio, how big is the gear on the stepper motor compared to the big gear of the table
    gear_ratio = tk.IntVar(value=config.gear_ratio)
    gear_ratio_label = ttk.Label(cap_win, text="Gear ratio", padding=[10])
    gear_ratio_label.grid(column=0, row=3)
    gear_ratio_entry = ttk.Entry(cap_win, text=gear_ratio)
    gear_ratio_entry.grid(column=1, row=3)

    #button to show minimal angle
    def min_angle_compute():
        spr = steps_per_rev.get()
        gr = gear_ratio.get()
        min_angle = 360/(spr*gr)
        min_angle_string = str(min_angle) + " °"
        min_angle_label.configure(text=min_angle_string)
        
    min_angle_button = ttk.Button(cap_win, text="Compute minimal angle")
    min_angle_button.grid(column=0, row=4, sticky="ew")
    min_angle_button.configure(command=min_angle_compute)

    min_angle_label = ttk.Label(cap_win)
    min_angle_label.grid(column=1, row=4)

    #desired angle input, beware that is has to be a multiple of the minimal angle
    des_angle = tk.DoubleVar(value=config.angstep)
    des_angle_label = ttk.Label(cap_win, text="Desired angle per step", padding=[10])
    des_angle_label.grid(column=0, row=5)
    des_angle_entry = ttk.Entry(cap_win, text=des_angle)
    des_angle_entry.grid(column=1, row=5)

    #show steps for full revolution
    def steps_per_angle():
        da = des_angle.get()
        spr = steps_per_rev.get()
        gr = gear_ratio.get()
        min_angle = 360/(spr*gr)
        steps_per_angle = int(da/min_angle)
        steps_per_angle_string = str(steps_per_angle) + " Steps per desired angle"
        steps_per_angle_label.configure(text=steps_per_angle_string)

    steps_per_angle_button = ttk.Button(cap_win, text="Compute steps per desired angle", padding=[10])
    steps_per_angle_button.grid(column=0, row=6, sticky="ew")
    steps_per_angle_button.configure(command=steps_per_angle)

    steps_per_angle_label = ttk.Label(cap_win, padding=[10])
    steps_per_angle_label.grid(column=1, row=6)
        
    proj_win_text = tk.Text(cap_win, height = 1, width = 20)
    proj_win_text.insert(tk.END, config.crosshair_path)
    proj_win_text.grid(column=1, row=7, sticky="ew")
    #button to summon the projection window
    def summ_proj_win():
        global proj_win
        proj_win = tk.Toplevel()
        proj_win.title("Projection Window")
        proj_win.geometry("600x600")
        img = ImageTk.PhotoImage(Image.open(config.crosshair_path))
        global proj_img_label
        proj_img_label = ttk.Label(proj_win, image=img)
        proj_img_label.image = img
        proj_img_label.place(relx=0.5, rely=0.5, anchor='center')
        
        

    proj_win_button = ttk.Button(cap_win, text="Summon Projection Window", padding=[10])
    proj_win_button.grid(column=0, row=7, sticky="ew")
    proj_win_button.configure(command=summ_proj_win)

    





    #function to change pattern and make images
    def projection_loop():
        da = des_angle.get()
        spr = steps_per_rev.get()
        gr = gear_ratio.get()
        min_angle = 360/(spr*gr)
        steps_per_angle = int(da/min_angle)
        target_folder = filedialog.askdirectory()
        da = des_angle.get()
        fit_des_ang_in_circ = int(np.ceil(360/da)) 
        position = int(pos_start_entry.get())
        for i in range(0,fit_des_ang_in_circ,1):
            for j in range(0,len(pattern_file_list),1):
                img2= ImageTk.PhotoImage(pattern_file_list[j])
                proj_img_label.configure(image=img2)
                proj_img_label.image = img2
                proj_img_label.update()
                cam_capture(target_folder, position, j)
            
            position = position + 1
            turntable(steps_per_angle)
            
        
        


    start_projection_button = ttk.Button(cap_win, text="Start Projection", padding=[10])
    start_projection_button.grid(column=0, row=8, sticky="ew")
    start_projection_button.configure(command=projection_loop)
    def on_close():
        global cap_win_state
        cap_win_state = False
        cap_win.destroy()
    cap_win.protocol("WM_DELETE_WINDOW", on_close)
 
def toggle_cal_window():
    global cal_win_state
    if not cal_win_state:
         calib_window()
         cal_win_state = True
def calib_window():
    cal_disp = tk.Toplevel(root)
    cal_disp.title("Camera Calibration")
    cal_disp.geometry('330x190')
    cal_disp.focus_force()
    cal_disp.resizable(width=False, height=False)
    def on_close():
        global cal_win_state
        cal_win_state = False
        cal_disp.destroy()
    cal_disp.protocol("WM_DELETE_WINDOW", on_close) 
    img_lbl = tk.Label(cal_disp, text = "Image Folder:")
    img_txt = tk.Text(cal_disp, height = 1, width = 20)
    img_txt.insert(tk.END, config.calib_img)
    img_lbl.grid(sticky = 'E',row = 0,column = 0)
    img_txt.grid(row = 0, column = 1)
    def img_btn_click():
        folder_path = filedialog.askdirectory()
        cal_disp.focus_force()
        img_txt.delete('1.0', tk.END)
        img_txt.insert('1.0', folder_path + "/")
    img_btn = tk.Button(cal_disp, text = "Browse", command = img_btn_click)
    img_btn.grid(row = 0, column = 2)
     
    left_lbl = tk.Label(cal_disp, text = "Left Indicator:")
    left_txt = tk.Text(cal_disp, height = 1, width = 20)
    left_txt.insert(tk.END, config.calib_left)
    left_lbl.grid(sticky = 'E',row = 1,column = 0)
    left_txt.grid(row = 1, column = 1) 
     
    right_lbl = tk.Label(cal_disp, text = "Right Indicator:")
    right_txt = tk.Text(cal_disp, height = 1, width = 20)
    right_txt.insert(tk.END, config.calib_right)
    right_lbl.grid(sticky = 'E',row = 2,column = 0)
    right_txt.grid(row = 2, column = 1)
    

    target_lbl = tk.Label(cal_disp, text = "Result Folder:")
    target_txt = tk.Text(cal_disp, height = 1, width = 20)
    target_txt.insert(tk.END, config.calib_target)
    target_lbl.grid(sticky = 'E',row = 3,column = 0)
    target_txt.grid(row = 3, column = 1)
    def target_btn_click():
        folder_path = filedialog.askdirectory()
        cal_disp.focus_force()
        target_txt.delete('1.0', tk.END)
        target_txt.insert('1.0', folder_path + "/")
    target_btn = tk.Button(cal_disp, text = "Browse", command = target_btn_click)
    target_btn.grid(row =3, column = 2)
     
    row_lbl = tk.Label(cal_disp, text = "Rows:")
    row_txt = tk.Text(cal_disp, height = 1, width = 20)
    row_txt.insert(tk.END, config.calib_rows)
    row_lbl.grid(sticky = 'E',row = 4, column = 0)
    row_txt.grid(row = 4, column = 1)
     
    col_lbl = tk.Label(cal_disp, text = "Columns:")
    col_txt = tk.Text(cal_disp, height = 1, width = 20)
    col_txt.insert(tk.END, config.calib_columns)
    col_lbl.grid(sticky = 'E',row = 5, column = 0)
    col_txt.grid(row =5, column = 1)
     
    sca_lbl = tk.Label(cal_disp, text = "Scale Length:")
    sca_txt = tk.Text(cal_disp, height = 1, width = 20)
    sca_txt.insert(tk.END, config.calib_scale)
    sca_lbl.grid(sticky = 'E',row = 6, column = 0)
    sca_txt.grid(row = 6, column = 1)
    def cal_check():
        error_flag = False
         
        input_chk =  img_txt.get('1.0',tk.END).rstrip()
        if (input_chk[-1] != "/"):
            tk.messagebox.showerror("Invalid Input", "Image Folder must end in '/'")
            error_flag = True
        indL_chk = left_txt.get('1.0',tk.END).rstrip()
        indR_chk = right_txt.get('1.0',tk.END).rstrip()
        if not error_flag:
            imgL_check = []
            imgR_check = []
            imgFull = []
            for file in os.listdir(input_chk):
                imgFull.append(file)
            for i in imgFull:
                if indL_chk in i:
                    imgL_check.append(i)
                elif indR_chk in i:
                    imgR_check.append(i)
            if len(indL_chk) != len(indR_chk):
                tk.messagebox.showerror("Invalid Image Count", "Specified left and right identifiers result in mismatched image count")
                error_flag = True
        target_chk = target_txt.get('1.0',tk.END).rstrip()
        if (target_chk[-1] != "/"):
            tk.messagebox.showerror("Invalid Input", "Calibration Result Folder must end in '/'")
            error_flag = True
        row_chk = row_txt.get('1.0',tk.END).rstrip()
        try:
            value = int(row_chk)
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Rows value must be an integer.")
            error_flag = True
        col_chk = col_txt.get('1.0',tk.END).rstrip()
        try:
            value = int(col_chk)
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Columns value must be an integer.")
            error_flag = True
        sca_chk = sca_txt.get('1.0',tk.END).rstrip()
        try:
            value = float(sca_chk)
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Scale value must be a float.")
            error_flag = True
        return error_flag
    def calst_btn_click():
        if not cal_check():
            config.calib_img = img_txt.get('1.0',tk.END).rstrip()
            config.calib_left = left_txt.get('1.0',tk.END).rstrip()
            config.calib_right = right_txt.get('1.0',tk.END).rstrip()
            config.calib_target = target_txt.get('1.0',tk.END).rstrip()
            
            config.calib_rows = int(row_txt.get('1.0',tk.END).rstrip())
            config.calib_columns = int(col_txt.get('1.0',tk.END).rstrip())
            config.calib_scale = float(sca_txt.get('1.0',tk.END).rstrip())
            mtx1, mtx2, dist_1, dist_2, R, T, E, F = scr.calibrate_cameras(config.calib_img, config.calib_left, config.calib_right, "", 
                                                                           config.calib_rows, config.calib_columns, 
                                                                           config.calib_scale)
            if mtx1 is not None:
                scr.fill_mtx_dir(config.calib_target, mtx1, mtx2, F, E, dist_1, dist_2, R, T)
    calst_btn = tk.Button(cal_disp, text = "Calibrate", command = calst_btn_click)
    calst_btn.grid(row = 7, column = 1)
    def cnc_btn_click():
        global cal_win_state
        cal_win_state = False
        cal_disp.destroy()
    cnc_btn = tk.Button(cal_disp, text = "Cancel", command = cnc_btn_click)
    cnc_btn.grid(row = 7, column = 2)
    def set_btn_click():
        if (not cal_check):
            config.calib_img = img_txt.get('1.0',tk.END).rstrip()
            config.calib_left = left_txt.get('1.0',tk.END).rstrip()
            config.calib_right = right_txt.get('1.0',tk.END).rstrip()
            config.calib_target = target_txt.get('1.0',tk.END).rstrip()
            config.calib_scale = sca_txt.get('1.0',tk.END).rstrip()
            config.calib_rows = row_txt.get('1.0',tk.END).rstrip()
            config.calib_columns = col_txt.get('1.0',tk.END).rstrip()
            config.make_config()
    set_btn = tk.Button(cal_disp, text = "Set Defaults", command = set_btn_click)
    set_btn.grid(row = 7, column = 0)
#MAIN UI
rec_btn = tk.Button(root, text = "Reconstruction", command  = toggle_rec_window)
rec_btn.pack()
cal_btn = tk.Button(root, text = "Camera Calibration", command = toggle_cal_window)
cal_btn.pack()
cap_btn = tk.Button(root, text = "Image Capture", command = toggle_cap_window)
cap_btn.pack()
root.geometry('400x200')
root.resizable(width=False, height=False)
root.focus_force()
root.mainloop()