# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:30:36 2023

@author: myuey
"""

import tkinter
from tkinter import filedialog
import confighandler as chand
import ncc_core as ncc
import os
global config
version = 1.432
config = chand.ConfigHandler(version)
config.load_config()
startup_cycle = True
#create window
root = tkinter.Tk()
root.title("3D Stereo Reconstruction -MG- FSU Jena - v" + str(version))
root.geometry('570x280')
root.resizable(width=False, height=False)
root.focus_force()

root.columnconfigure(2, minsize=10)

#output filebox
out_lbl = tkinter.Label(root, text = "Output File:")
out_lbl.grid(sticky="E", row=0, column=0)
out_txt = tkinter.Text(root, height=1, width=35)
out_txt.insert(tkinter.END, config.output)
out_txt.grid(row=0, column=1)
#variables
mat_fold = tkinter.StringVar(root)
imgL_fold = tkinter.StringVar(root)
imgR_fold = tkinter.StringVar(root)
multi_bool = tkinter.BooleanVar(root)
multi_bool.set(False)
multi_num = tkinter.IntVar(root)
loaf_bool = tkinter.BooleanVar(root)
loaf_bool.set(False)
savef_bool = tkinter.BooleanVar(root)
savef_bool.set(False)
precise_bool = tkinter.BooleanVar(root)
precise_bool.set(config.precise > 0)
speed_bool = tkinter.BooleanVar(root)
speed_bool.set(config.speed_mode > 0)
data_bool = tkinter.BooleanVar(root)
data_bool.set(config.data_out > 0)

#matrix folder location
mat_lbl = tkinter.Label(root, text = "Matrices:")
mat_lbl.grid(sticky="E",row = 1, column = 0)
mat_txt = tkinter.Text(root, height = 1, width = 35)
mat_txt.insert(tkinter.END, config.mat_folder)
mat_txt.grid(sticky="E", row = 1, column = 1)
def mat_btn_click():
    folder_path = filedialog.askdirectory()
    mat_fold.set(folder_path + "/")
    mat_txt.delete('1.0', tkinter.END)
    mat_txt.insert('1.0', folder_path + "/")
mat_btn = tkinter.Button(root, text = "Browse", command = mat_btn_click)
mat_btn.grid(sticky="W",row = 1, column = 2)
#images_L location
imgL_lbl = tkinter.Label(root, text = "Left Images:")
imgL_lbl.grid(sticky="E", row = 2, column = 0)
imgL_txt = tkinter.Text(root, height = 1, width = 35)
imgL_txt.insert(tkinter.END, config.left_folder)
imgL_txt.grid(row = 2, column = 1)
def imgL_btn_click():
    folder_path = filedialog.askdirectory()
    imgL_fold.set(folder_path + "/")
    imgL_txt.delete('1.0', tkinter.END)
    imgL_txt.insert('1.0', folder_path + "/")
imgL_btn = tkinter.Button(root, text = "Browse", command = imgL_btn_click)
imgL_btn.grid(sticky="W",row = 2, column = 2)
#images_R location
imgR_lbl = tkinter.Label(root, text = "Right Images:")
imgR_lbl.grid(sticky="E", row = 3, column = 0)
imgR_txt = tkinter.Text(root, height = 1, width = 35)
imgR_txt.insert(tkinter.END, config.right_folder)
imgR_txt.grid(row = 3, column = 1)
def imgR_btn_click():
    folder_path = filedialog.askdirectory()
    imgR_fold.set(folder_path + "/")
    imgR_txt.delete('1.0', tkinter.END)
    imgR_txt.insert('1.0', folder_path + "/")
imgR_btn = tkinter.Button(root, text = "Browse", command = imgR_btn_click)
imgR_btn.grid(sticky="W",row = 3, column = 2)
#interpolation points input
interp_lbl = tkinter.Label(root, text = "Interpolations:")
interp_lbl.grid(sticky="E", row = 4, column = 0)
interp_txt = tkinter.Text(root, height = 1, width = 35)
interp_txt.insert(tkinter.END, config.interp)
interp_txt.grid(row = 4, column = 1)
#offset value input
ofsXL_lbl = tkinter.Label(root, text = "Offset X Left:")
ofsXL_lbl.grid(sticky="E", row = 5, column = 0)
ofsXL_txt = tkinter.Text(root, height = 1, width = 35)
ofsXL_txt.insert(tkinter.END, config.x_offset_L)
ofsXL_txt.grid(row = 5, column = 1)

ofsXR_lbl = tkinter.Label(root, text = "Offset X Right:")
ofsXR_lbl.grid(sticky="E", row = 6, column = 0)
ofsXR_txt = tkinter.Text(root, height = 1, width = 35)
ofsXR_txt.insert(tkinter.END, config.x_offset_R)
ofsXR_txt.grid(row = 6, column = 1)

ofsYT_lbl = tkinter.Label(root, text = "Offset Y Top:")
ofsYT_lbl.grid(sticky="E", row = 7, column = 0)
ofsYT_txt = tkinter.Text(root, height = 1, width = 35)
ofsYT_txt.insert(tkinter.END, config.y_offset_T)
ofsYT_txt.grid(row = 7, column = 1)

ofsYB_lbl = tkinter.Label(root, text = "Offset Y Bottom:")
ofsYB_lbl.grid(sticky="E", row = 8, column = 0)
ofsYB_txt = tkinter.Text(root, height = 1, width = 35)
ofsYB_txt.insert(tkinter.END, config.y_offset_B)
ofsYB_txt.grid(row = 8, column = 1)
def check_folder(path):
    contents = os.listdir(path)
    for item in contents:
        if not os.path.isdir(os.path.join(path, item)):
            return False
    return True
#Error messages for invalid entries
def entry_check_main(isMap = False):
    error_flag = False
    verif_left = False
    verif_right = False
    if (isMap):
        pass
    else:
        pass
    mat_fol_chk = mat_txt.get('1.0', tkinter.END).rstrip()
    if (mat_fol_chk[-1] != "/"):
        tkinter.messagebox.showerror("Invalid Input", "Matrix Folder must end in '/'")
        error_flag = True
    elif(not os.path.isdir(mat_fol_chk)):
        tkinter.messagebox.showerror("Folder Not Found", "Specified Matrix Folder '" + mat_fol_chk +
                                      "' not found.")
        error_flag = True
    
    imgL_chk = imgL_txt.get('1.0', tkinter.END).rstrip()
    if (imgL_chk[-1] != "/"):
        tkinter.messagebox.showerror("Invalid Input", "Left Images Folder must end in  '/'")
        error_flag = True
    elif(not os.path.isdir(imgL_chk)):
        tkinter.messagebox.showerror("Folder Not Found", "Specified Left Images Folder '" + imgL_chk +
                                      "' not found.")
        error_flag = True
    elif not check_folder(imgL_chk) and multi_bool.get():
        tkinter.messagebox.showerror("Folder Error", "Multiple Runs mode selected but specified Left Images Folder '" + imgL_chk +
                                      "' does not contain only folders.")
        error_flag = True
    elif check_folder(imgL_chk) and not multi_bool.get():
        tkinter.messagebox.showerror("Folder Error", "Multiple Runs mode not selected but specified Left Images Folder '" + imgL_chk +
                                      "' contains only folders.")
        error_flag = True
    else:
        verif_left = True
    imgR_chk = imgR_txt.get('1.0', tkinter.END).rstrip()
    if (imgR_chk[-1] != "/"):
        tkinter.messagebox.showerror("Invalid Input", "Right Images Folder must end in  '/'")
        error_flag = True
    elif(not os.path.isdir(imgR_chk)):
        tkinter.messagebox.showerror("Folder Not Found", "Specified Right Images Folder '" + imgR_chk +
                                      "' not found.")
    
        error_flag = True
    elif not check_folder(imgR_chk) and multi_bool.get():
        tkinter.messagebox.showerror("Folder Error", "Multiple Runs mode selected but specified Right Images Folder '" + imgR_chk +
                                      "' does not contain only folders.")
        error_flag = True
    elif check_folder(imgR_chk) and not multi_bool.get():
        tkinter.messagebox.showerror("Folder Error", "Multiple Runs mode not selected but specified Right Images Folder '" + imgR_chk +
                                      "' contains only folders.")
        error_flag = True
    else:
        verif_right = True
    if(verif_left and verif_right):
        left_len = len(os.listdir(imgL_chk))
        right_len = len(os.listdir(imgR_chk))
        if left_len != right_len:
            tkinter.messagebox.showerror("Mismatched Image Source", "Number of directories in '" + imgL_chk + "' and '" + 
                                        imgR_chk + "' do not match.")
            error_flag = True
    interp_chk = interp_txt.get('1.0', tkinter.END).rstrip()
    try:
        value = int(interp_chk)
    except ValueError:
        tkinter.messagebox.showerror("Invalid Input", "Interpolations value must be integer")
        error_flag = True
    x_offL_chk = ofsXL_txt.get('1.0', tkinter.END).rstrip()
    try:
        value = int(x_offL_chk)
        if value == 0:
            tkinter.messagebox.showerror("Invalid Input", "X Offset value must be integer > 0")
            error_flag = True
    except ValueError:
        tkinter.messagebox.showerror("Invalid Input", "X Offset value must be integer")
        error_flag = True
    x_offR_chk = ofsXR_txt.get('1.0', tkinter.END).rstrip()
    try:
        value = int(x_offR_chk)
        if value == 0:
            tkinter.messagebox.showerror("Invalid Input", "X Offset value must be integer > 0")
            error_flag = True
    except ValueError:
        tkinter.messagebox.showerror("Invalid Input", "X Offset value must be integer")
        error_flag = True
    y_offT_chk = ofsYT_txt.get('1.0', tkinter.END).rstrip()
    try:
        value = int(y_offT_chk)
        if value == 0:
            tkinter.messagebox.showerror("Invalid Input", "Y Offset value must be integer > 0")
            error_flag = True
    except ValueError:
        tkinter.messagebox.showerror("Invalid Input", "Y Offset value must be integer > 0")
        error_flag = True
    y_offB_chk = ofsYB_txt.get('1.0', tkinter.END).rstrip()
    try:
        value = int(y_offB_chk)
        if value == 0:
            tkinter.messagebox.showerror("Invalid Input", "Y Offset value must be integer > 0")
            error_flag = True
    except ValueError:
        tkinter.messagebox.showerror("Invalid Input", "Y Offset value must be integer > 0")
        error_flag = True    
    return error_flag
#multi-recon checkbox

multi_box = tkinter.Checkbutton(root, text="Multiple Runs", variable=multi_bool)
multi_box.grid(sticky="W",row = 5, column = 3)

#speed checkbox
speed_box= tkinter.Checkbutton(root, text="Increase Speed", variable=speed_bool)
speed_box.grid(sticky="W",row = 6, column = 3)

#Full data checkbox
data_box= tkinter.Checkbutton(root, text="Data Out", variable=data_bool)
data_box.grid(sticky="W",row = 7, column = 3)
#start button
def st_btn_click(): 
    entry_chk = entry_check_main()
    if not entry_chk and not multi_bool.get():
        print("Creating Reconstruction")
        config.mat_folder = mat_txt.get('1.0', tkinter.END).rstrip()
        config.left_folder = imgL_txt.get('1.0', tkinter.END).rstrip()
        config.right_folder = imgR_txt.get('1.0', tkinter.END).rstrip()
        config.interp = int(interp_txt.get('1.0', tkinter.END).rstrip())
        config.x_offset_L = int(ofsXL_txt.get('1.0', tkinter.END).rstrip())
        config.x_offset_R = int(ofsXR_txt.get('1.0', tkinter.END).rstrip())
        config.y_offset_T = int(ofsYT_txt.get('1.0', tkinter.END).rstrip())
        config.y_offset_B = int(ofsYB_txt.get('1.0', tkinter.END).rstrip())
        config.output = out_txt.get('1.0', tkinter.END).rstrip()
        config.speed_mode = speed_bool.get()
        config.data_out = data_bool.get()
        ncc.run_cor(config)
    elif not entry_chk and multi_bool.get():
        print("Creating Reconstruction")
        config.mat_folder = mat_txt.get('1.0', tkinter.END).rstrip()
        left_base = imgL_txt.get('1.0', tkinter.END).rstrip()
        right_base = imgR_txt.get('1.0', tkinter.END).rstrip()
        config.speed_mode = speed_bool.get()
        config.interp = int(interp_txt.get('1.0', tkinter.END).rstrip())
        config.x_offset_L = int(ofsXL_txt.get('1.0', tkinter.END).rstrip())
        config.x_offset_R = int(ofsXR_txt.get('1.0', tkinter.END).rstrip())
        config.y_offset_T = int(ofsYT_txt.get('1.0', tkinter.END).rstrip())
        config.y_offset_B = int(ofsYB_txt.get('1.0', tkinter.END).rstrip())
        out_base = out_txt.get('1.0', tkinter.END).rstrip()
        config.speed_mode = speed_bool.get()
        config.data_out = data_bool.get()
        if "." in out_base:
            out_base = out_base.split(".", 1)[0]
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
st_btn = tkinter.Button(root, text = "Start Reconstruction", command = st_btn_click)
st_btn.grid(row = 10, column = 1)
#correlation map 
map_lbl = tkinter.Label(root, text = "Correlation Map File:")
map_lbl.grid(sticky="E", row = 9, column = 0)
map_txt = tkinter.Text(root, height = 1, width = 35)
map_txt.insert(tkinter.END, config.corr_map_name)
map_txt.grid(row = 9, column = 1)
def cor_map_btn_click():
    
    entry_chk = entry_check_main()
    if not entry_chk:
        print("Creating Correlation Map")
        config.mat_folder = mat_txt.get('1.0', tkinter.END).rstrip()
        config.left_folder = imgL_txt.get('1.0', tkinter.END).rstrip()
        config.right_folder = imgR_txt.get('1.0', tkinter.END).rstrip()
        config.interp = int(interp_txt.get('1.0', tkinter.END).rstrip())
        config.x_offset_L = int(ofsXL_txt.get('1.0', tkinter.END).rstrip())
        config.x_offset_R = int(ofsXR_txt.get('1.0', tkinter.END).rstrip())
        config.y_offset_T = int(ofsYT_txt.get('1.0', tkinter.END).rstrip())
        config.y_offset_B = int(ofsYB_txt.get('1.0', tkinter.END).rstrip())
        config.corr_map_name = map_txt.get('1.0', tkinter.END).rstrip()
        if(speed_bool.get()):
            config.speed_mode = 1
        else:
            config.speed_mode = 0
        ncc.run_cor(config, mapgen = True)
map_btn = tkinter.Button(root, text = "Create", command = cor_map_btn_click)
map_btn.grid(row = 9, column = 2)
#reset button
def rst_btn_click():
    config = chand.ConfigHandler(version)
    out_txt.delete('1.0', tkinter.END)
    out_txt.insert(tkinter.END, config.output)
    mat_txt.delete('1.0', tkinter.END)
    mat_txt.insert(tkinter.END, config.mat_folder)
    imgL_txt.delete('1.0', tkinter.END)
    imgL_txt.insert(tkinter.END, config.left_folder)
    imgR_txt.delete('1.0', tkinter.END)
    imgR_txt.insert(tkinter.END, config.right_folder)
    interp_txt.delete('1.0', tkinter.END)
    interp_txt.insert(tkinter.END, config.interp)
    ofsXL_txt.delete('1.0', tkinter.END)
    ofsXL_txt.insert(tkinter.END, config.x_offset_L)
    ofsXR_txt.delete('1.0', tkinter.END)
    ofsXR_txt.insert(tkinter.END, config.x_offset_R)
    ofsYT_txt.delete('1.0', tkinter.END)
    ofsYT_txt.insert(tkinter.END, config.y_offset_T)
    ofsYB_txt.delete('1.0', tkinter.END)
    ofsYB_txt.insert(tkinter.END, config.y_offset_B)
    map_txt.delete('1.0', tkinter.END)
    map_txt.insert(tkinter.END, config.corr_map_name)
rst_btn = tkinter.Button(root, text = "Reset", command = rst_btn_click)
rst_btn.grid(row = 2, column = 3)
#save all fields as default button - if field is empty, do not modify config
def cfg_btn_click(): 
    if not entry_check_main():
        config.output = out_txt.get('1.0',tkinter.END).rstrip()
        config.mat_folder = mat_txt.get('1.0', tkinter.END).rstrip()
        config.left_folder = imgL_txt.get('1.0', tkinter.END).rstrip()
        config.right_folder = imgR_txt.get('1.0', tkinter.END).rstrip()
    
        config.interp = int(interp_txt.get('1.0', tkinter.END).rstrip())
    
        config.x_offset_L = int(ofsXL_txt.get('1.0', tkinter.END).rstrip())
        config.x_offset_R = int(ofsXR_txt.get('1.0', tkinter.END).rstrip())
        config.y_offset_T = int(ofsYT_txt.get('1.0', tkinter.END).rstrip())
        config.y_offset_B = int(ofsYB_txt.get('1.0', tkinter.END).rstrip())
        config.corr_map_name = map_txt.get('1.0',tkinter.END).rstrip()
        config.make_config()   
    
cfg_btn = tkinter.Button(root, text = "Set Defaults", command = cfg_btn_click)
cfg_btn.grid(row = 1, column = 3)

#settings window
def set_window():
    set_disp = tkinter.Toplevel(root)
    set_disp.title("Settings")
    set_disp.geometry('410x280')
    set_disp.focus_force()
    set_disp.resizable(width=False, height=False)
    
    tmod_lbl = tkinter.Label(set_disp, text = "t-Vector Scale Factor:")
    tmod_txt = tkinter.Text(set_disp, height = 1, width = 20)
    tmod_txt.insert(tkinter.END, config.tmod)
    tmod_lbl.grid(row = 0, column = 0)
    tmod_txt.grid(row = 0, column = 1)
    
    tvec_lbl = tkinter.Label(set_disp, text = "t-Vector File:")
    tvec_txt = tkinter.Text(set_disp, height = 1, width = 20)
    tvec_txt.insert(tkinter.END, config.t_file)
    tvec_lbl.grid(row = 1, column = 0)
    tvec_txt.grid(row = 1, column = 1)
    
    Rmat_lbl = tkinter.Label(set_disp, text = "R-Matrix File:")
    Rmat_txt = tkinter.Text(set_disp, height = 1, width = 20)
    Rmat_txt.insert(tkinter.END, config.R_file)
    Rmat_lbl.grid(row = 2, column = 0)
    Rmat_txt.grid(row = 2, column = 1)
    
    lkp_lbl = tkinter.Label(set_disp, text = "Lineskips:")
    lkp_txt = tkinter.Text(set_disp, height = 1, width = 20)
    lkp_txt.insert(tkinter.END, config.skiprow)
    lkp_lbl.grid(row = 3, column = 0)
    lkp_txt.grid(row = 3, column = 1)
    
    kl_lbl = tkinter.Label(set_disp, text = "Left Camera Matrix File:")
    kl_txt = tkinter.Text(set_disp, height = 1, width = 20)
    kl_txt.insert(tkinter.END, config.kL_file)
    kl_lbl.grid(row = 4, column = 0)
    kl_txt.grid(row = 4, column = 1)
    
    kr_lbl = tkinter.Label(set_disp, text = "Right Camera Matrix File:")
    kr_txt = tkinter.Text(set_disp, height = 1, width = 20)
    kr_txt.insert(tkinter.END, config.kR_file)
    kr_lbl.grid(row = 5, column = 0)
    kr_txt.grid(row = 5, column = 1)
    
    f_lbl = tkinter.Label(set_disp, text = "Fundamental Matrix File:")
    f_txt = tkinter.Text(set_disp, height = 1, width = 20)
    f_txt.insert(tkinter.END, config.f_file)
    f_lbl.grid(row = 6, column = 0)
    f_txt.grid(row = 6, column = 1)
    
    delim_lbl = tkinter.Label(set_disp, text = "Delimiter:")
    delim_txt = tkinter.Text(set_disp, height = 1, width = 20)
    delim_txt.insert(tkinter.END, config.delim)
    delim_lbl.grid(row = 7, column = 0)
    delim_txt.grid(row = 7, column = 1)
    
    thr_lbl = tkinter.Label(set_disp, text = "Correlation Threshold:")
    thr_txt = tkinter.Text(set_disp, height = 1, width = 20)
    thr_txt.insert(tkinter.END, config.thresh)
    thr_lbl.grid(row = 8, column = 0)
    thr_txt.grid(row = 8, column = 1)
    
    msk_lbl = tkinter.Label(set_disp, text = "Mask Threshold:")
    msk_txt = tkinter.Text(set_disp, height = 1, width = 20)
    msk_txt.insert(tkinter.END, config.mask_thresh)
    msk_lbl.grid(row = 9, column = 0)
    msk_txt.grid(row = 9, column = 1)
    
    spd_lbl = tkinter.Label(set_disp, text = "Speed Interval:")
    spd_txt = tkinter.Text(set_disp, height = 1, width = 20)
    spd_txt.insert(tkinter.END, config.speed_interval)
    spd_lbl.grid(row = 10, column = 0)
    spd_txt.grid(row = 10, column = 1)
    
    flo_box = tkinter.Checkbutton(set_disp, text="Load F Matrix", variable=loaf_bool)
    flo_box.grid(row = 6, column =2)
    flo_box = tkinter.Checkbutton(set_disp, text="Save F Matrix", variable=savef_bool)
    flo_box.grid(row = 7, column =2)
    #Precision checkbox
    precise_box = tkinter.Checkbutton(set_disp, text="Counter Skew", variable=precise_bool)
    precise_box.grid(row = 0, column = 2)
    def entry_check_settings():
        error_flag = False
        mat_fold = mat_txt.get('1.0', tkinter.END).rstrip()
        tmod_chk = tmod_txt.get('1.0',tkinter.END).rstrip()
        try:
            value = float(tmod_chk)
        except ValueError:
            tkinter.messagebox.showerror("Invalid Input", "t-vector scale factor value must be float.")
            error_flag = True
        tvec_chk = tvec_txt.get('1.0',tkinter.END).rstrip()
        if (not tvec_chk.endswith(".txt")):
            tkinter.messagebox.showerror("Invalid Input", "t-vector file type must be .txt.")
            error_flag = True
        elif(not os.path.isfile(mat_fold + tvec_chk)):
            tkinter.messagebox.showerror("File Not Found", "Specified t-vector file '" + mat_fold + tvec_chk +
                                         "' not found.")
            error_flag = True
        Rmat_chk = Rmat_txt.get('1.0',tkinter.END).rstrip()
        if (not Rmat_chk.endswith(".txt")):
            tkinter.messagebox.showerror("Invalid Input", "R-matrix file type must be .txt.")
            error_flag = True
            
        elif(not os.path.isfile(mat_fold + Rmat_chk)):
            tkinter.messagebox.showerror("File Not Found", "Specified R-Matrix file '" + mat_fold + Rmat_chk +
                                         "' not found.")
            error_flag = True
        f_chk = f_txt.get('1.0',tkinter.END).rstrip()
        if (not f_chk.endswith(".txt")):
            tkinter.messagebox.showerror("Invalid Input", "Fundamental matrix file type must be .txt.")
            error_flag = True
        skiprow_chk = lkp_txt.get('1.0',tkinter.END).rstrip()
        try:
            value = int(skiprow_chk)
        except ValueError:
            tkinter.messagebox.showerror("Invalid Input", "Lineskips value must be an integer.")
            error_flag = True
        kL_file_chk = kl_txt.get('1.0',tkinter.END).rstrip()
        if (not kL_file_chk.endswith(".txt")):
            tkinter.messagebox.showerror("Invalid Input", "Left Camera Matrix file type must be .txt.")
            error_flag = True
        elif(not os.path.isfile(mat_fold + kL_file_chk)):
            tkinter.messagebox.showerror("File Not Found", "Specified Left Camera Matrix file '" + mat_fold + kL_file_chk +
                                         "' not found.")
            error_flag = True
        kR_file_chk = kr_txt.get('1.0',tkinter.END).rstrip()
        if (not kR_file_chk.endswith(".txt")):
            tkinter.messagebox.showerror("Invalid Input", "Right Camera Matrix file type must be .txt.")
            error_flag = True
        elif(not os.path.isfile(mat_fold + kR_file_chk)):
            tkinter.messagebox.showerror("File Not Found", "Specified Right Camera Matrix file '" + mat_fold + kR_file_chk +
                                         "' not found.")
            error_flag = True
        thresh_chk = thr_txt.get('1.0',tkinter.END).rstrip()
        try:
            value = float(thresh_chk)
        except ValueError:
            tkinter.messagebox.showerror("Invalid Input", "Correlation Threshold value must be a float.")
            error_flag = True
        msk_chk = msk_txt.get('1.0',tkinter.END).rstrip()
        try:
            value = int(msk_chk)
        except ValueError:
            tkinter.messagebox.showerror("Invalid Input", "Mask Threshold value must be an integer.")
            error_flag = True
        if (loaf_bool.get() and savef_bool.get()):
            tkinter.messagebox.showerror("Invalid Input", "Saving F matrix is incompatible with loading F matrix at the same time.")
            error_flag = True
        spd_chk = spd_txt.get('1.0',tkinter.END).rstrip()   
        try:
            value = int(spd_chk)
        except ValueError:
            tkinter.messagebox.showerror("Invalid Input", "Speed Interval value must be an integer.")
            error_flag = True
        return error_flag
    def cnc_btn_click():
        set_disp.destroy()
    cnc_btn = tkinter.Button(set_disp, text = "Cancel", command = cnc_btn_click)
    def ok_btn_click():
        if not entry_check_settings():
            config.tmod = float(tmod_txt.get('1.0',tkinter.END).rstrip())
            config.t_file = tvec_txt.get('1.0',tkinter.END).rstrip()
            config.R_file = Rmat_txt.get('1.0',tkinter.END).rstrip()
            config.skiprow = int(lkp_txt.get('1.0',tkinter.END).rstrip())
            config.kL_file = kl_txt.get('1.0',tkinter.END).rstrip()
            config.kR_file = kr_txt.get('1.0',tkinter.END).rstrip()
            config.f_file = f_txt.get('1.0',tkinter.END).rstrip()
            if(delim_txt.get('1.0',tkinter.END).rstrip() == ""):
                config.delim = " "
            else:
                config.delim = delim_txt.get('1.0',tkinter.END).rstrip()
            config.thresh = float(thr_txt.get('1.0',tkinter.END).rstrip())
            config.mask_thresh = int(msk_txt.get('1.0',tkinter.END).rstrip())
            if(loaf_bool.get()):
                config.f_load = 1
            else:
                config.f_load = 0
            if(savef_bool.get()):
                config.f_save = 1 
            else:
                config.f_save = 0
            if(precise_bool.get()):
                config.precise = 1
            else:
                config.precise = 0
            
            config.speed_interval = int(spd_txt.get('1.0',tkinter.END).rstrip())
            set_disp.destroy()
    ok_btn = tkinter.Button(set_disp, text = "OK", command = ok_btn_click)
    
    cnc_btn.grid(row = 11, column = 0)
    ok_btn.grid(row = 11,column = 1)

set_btn = tkinter.Button(root, text = "Settings", command = set_window)
set_btn.grid(row = 3, column = 3)
'''
#calibration window using calibration grid
def calib_window():
    cal_disp = tkinter.Toplevel(root)
    cal_disp.title("Grid Calibration")
    cal_disp.geometry('400x250')
    cal_disp.focus_force()
    cal_disp.resizable(width=False, height=False)
    
    left_lbl = tkinter.Label(cal_disp, text = "Left Image Folder")
    left_txt = tkinter.Text(cal_disp, height = 1, width = 20)
    left_txt.insert(tkinter.END, config.left_calib)
    
    right_lbl = tkinter.Label(cal_disp, text = "Right Image Folder")
    right_txt = tkinter.Text(cal_disp, height = 1, width = 20)
    right_txt.insert(tkinter.END, config.right_calib)
    
    target_lbl = tkinter.Label()
    target_txt = tkinter.Text()
    target_txt.insert(tkinter.END, config.target)
    def cal_check():
        pass
    def start_btn_click():
        pass
    start_btn = tkinter.Button(cal_disp, text = "Calibrate", command = start_btn_click)
    
    def cnc_btn_click():
        cal_disp.destroy()
    cnc_btn = tkinter.Button(cal_disp, text = "Cancel", command = cnc_btn_click)
cal_btn = tkinter.Button(root, text = "Camera Calibration", command = calib_window)
cal_btn.grid(row = 4, column = 5)
'''
root.mainloop()

    
