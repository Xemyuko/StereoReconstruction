# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:30:36 2023

@author: myuey
"""

import tkinter
from tkinter import filedialog
import confighandler as chand
import ncc_cor_core as ncc
import os
global config
config = chand.ConfigHandler()
config.load_config()
startup_cycle = True
#create window
root = tkinter.Tk()
root.title("3D Stereo Reconstruction -MG- FSU Jena - v1.2")
root.geometry('600x200')
root.resizable(width=False, height=False)
root.focus_force()

root.columnconfigure(2, minsize=10)
#root.rowconfigure(0, minsize=50)

#output filebox
out_lbl = tkinter.Label(root, text = "Output File:")
out_lbl.grid(row=0, column=0)
out_txt = tkinter.Text(root, height=1, width=35)
out_txt.insert(tkinter.END, config.output)
out_txt.grid(row=0, column=1)
#folder variables
mat_fold = tkinter.StringVar(root)
imgL_fold = tkinter.StringVar(root)
imgR_fold = tkinter.StringVar(root)
multi_bool = tkinter.BooleanVar(root)
multi_bool.set(False)
multi_num = tkinter.IntVar(root)
#matrix folder location
mat_lbl = tkinter.Label(root, text = "Matrices:")
mat_lbl.grid(row = 1, column = 0)
mat_txt = tkinter.Text(root, height = 1, width = 35)
mat_txt.insert(tkinter.END, config.mat_folder)
mat_txt.grid(row = 1, column = 1)
def mat_btn_click():
    folder_path = filedialog.askdirectory()
    mat_fold.set(folder_path + "/")
    mat_txt.delete('1.0', tkinter.END)
    mat_txt.insert('1.0', folder_path + "/")
mat_btn = tkinter.Button(root, text = "Browse", command = mat_btn_click)
mat_btn.grid(row = 1, column = 2)
#images_L location
imgL_lbl = tkinter.Label(root, text = "Left Images:")
imgL_lbl.grid(row = 2, column = 0)
imgL_txt = tkinter.Text(root, height = 1, width = 35)
imgL_txt.insert(tkinter.END, config.left_folder)
imgL_txt.grid(row = 2, column = 1)
def imgL_btn_click():
    folder_path = filedialog.askdirectory()
    imgL_fold.set(folder_path + "/")
    imgL_txt.delete('1.0', tkinter.END)
    imgL_txt.insert('1.0', folder_path + "/")
imgL_btn = tkinter.Button(root, text = "Browse", command = imgL_btn_click)
imgL_btn.grid(row = 2, column = 2)
#images_R location
imgR_lbl = tkinter.Label(root, text = "Right Images:")
imgR_lbl.grid(row = 3, column = 0)
imgR_txt = tkinter.Text(root, height = 1, width = 35)
imgR_txt.insert(tkinter.END, config.right_folder)
imgR_txt.grid(row = 3, column = 1)
def imgR_btn_click():
    folder_path = filedialog.askdirectory()
    imgR_fold.set(folder_path + "/")
    imgR_txt.delete('1.0', tkinter.END)
    imgR_txt.insert('1.0', folder_path + "/")
imgR_btn = tkinter.Button(root, text = "Browse", command = imgR_btn_click)
imgR_btn.grid(row = 3, column = 2)
#interpolation points input
interp_lbl = tkinter.Label(root, text = "Interpolations:")
interp_lbl.grid(row = 4, column = 0)
interp_txt = tkinter.Text(root, height = 1, width = 35)
interp_txt.insert(tkinter.END, config.interp)
interp_txt.grid(row = 4, column = 1)
#offset value input
ofsX_lbl = tkinter.Label(root, text = "Offset X:")
ofsX_lbl.grid(row = 5, column = 0)
ofsX_txt = tkinter.Text(root, height = 1, width = 35)
ofsX_txt.insert(tkinter.END, config.x_offset)
ofsX_txt.grid(row = 5, column = 1)
ofsY_lbl = tkinter.Label(root, text = "Offset Y:")
ofsY_lbl.grid(row = 6, column = 0)
ofsY_txt = tkinter.Text(root, height = 1, width = 35)
ofsY_txt.insert(tkinter.END, config.y_offset)
ofsY_txt.grid(row = 6, column = 1)
def check_folder(path):
    contents = os.listdir(path)
    for item in contents:
        if not os.path.isdir(os.path.join(path, item)):
            return False
    return True
#Error messages for invalid entries
def entry_check_main():
    error_flag = False
    verif_left = False
    verif_right = False
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
    x_off_chk = ofsX_txt.get('1.0', tkinter.END).rstrip()
    try:
        value = int(x_off_chk)
    except ValueError:
        tkinter.messagebox.showerror("Invalid Input", "X Offset value must be integer")
        error_flag = True
    y_off_chk = ofsY_txt.get('1.0', tkinter.END).rstrip()
    try:
        value = int(y_off_chk)
    except ValueError:
        tkinter.messagebox.showerror("Invalid Input", "Y Offset value must be integer")
        error_flag = True
        
    return error_flag
#multi-recon checkbox

multi_box = tkinter.Checkbutton(root, text="Multiple Runs", variable=multi_bool)
multi_box.grid(row = 4, column = 2)
#start button
def st_btn_click(): 
    entry_chk = entry_check_main()
    if not entry_chk and not multi_bool.get():
        config.mat_folder = mat_txt.get('1.0', tkinter.END).rstrip()
        config.left_folder = imgL_txt.get('1.0', tkinter.END).rstrip()
        config.right_folder = imgR_txt.get('1.0', tkinter.END).rstrip()
        config.interp = int(interp_txt.get('1.0', tkinter.END).rstrip())
        config.x_offset = int(ofsX_txt.get('1.0', tkinter.END).rstrip())
        config.y_offset = int(ofsY_txt.get('1.0', tkinter.END).rstrip())
        config.output = out_txt.get('1.0', tkinter.END).rstrip()
        root.after(20, ncc.run_cor_lin(config))
    elif not entry_chk and multi_bool.get():
        config.mat_folder = mat_txt.get('1.0', tkinter.END).rstrip()
        left_base = imgL_txt.get('1.0', tkinter.END).rstrip()
        right_base = imgR_txt.get('1.0', tkinter.END).rstrip()
        
        config.interp = int(interp_txt.get('1.0', tkinter.END).rstrip())
        config.x_offset = int(ofsX_txt.get('1.0', tkinter.END).rstrip())
        config.y_offset = int(ofsY_txt.get('1.0', tkinter.END).rstrip())
        out_base = out_txt.get('1.0', tkinter.END).rstrip()
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
                ncc.run_cor_lin(config)
            counter+=1
st_btn = tkinter.Button(root, text = "Start", command = st_btn_click)
st_btn.grid(row = 7, column = 1)

#reset button
def rst_btn_click():
    config.load_config()
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
    ofsX_txt.delete('1.0', tkinter.END)
    ofsX_txt.insert(tkinter.END, config.x_offset)
    ofsY_txt.delete('1.0', tkinter.END)
    ofsY_txt.insert(tkinter.END, config.x_offset)
    
rst_btn = tkinter.Button(root, text = "Reset", command = rst_btn_click)
rst_btn.grid(row = 2, column = 5)
#save all fields as default button - if field is empty, do not modify config
def cfg_btn_click(): 
    if not entry_check_main():
        config.output = out_txt.get('1.0',tkinter.END).rstrip()
        config.mat_folder = mat_txt.get('1.0', tkinter.END).rstrip()
        config.left_folder = imgL_txt.get('1.0', tkinter.END).rstrip()
        config.right_folder = imgR_txt.get('1.0', tkinter.END).rstrip()
    
        config.interp = int(interp_txt.get('1.0', tkinter.END).rstrip())
    
        config.x_offset = int(ofsX_txt.get('1.0', tkinter.END).rstrip())
    
        config.y_offset = int(ofsY_txt.get('1.0', tkinter.END).rstrip())
        
        config.make_config()   
    
cfg_btn = tkinter.Button(root, text = "Set Defaults", command = cfg_btn_click)
cfg_btn.grid(row = 1, column = 5)
#settings window
def set_window():
    set_disp = tkinter.Toplevel(root)
    set_disp.title("Settings")
    set_disp.geometry('350x230')
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
    
    delim_lbl = tkinter.Label(set_disp, text = "Delimiter:")
    delim_txt = tkinter.Text(set_disp, height = 1, width = 20)
    delim_txt.insert(tkinter.END, config.delim)
    delim_lbl.grid(row = 6, column = 0)
    delim_txt.grid(row = 6, column = 1)
    
    thr_lbl = tkinter.Label(set_disp, text = "Correlation Threshold:")
    thr_txt = tkinter.Text(set_disp, height = 1, width = 20)
    thr_txt.insert(tkinter.END, config.thresh)
    thr_lbl.grid(row = 7, column = 0)
    thr_txt.grid(row = 7, column = 1)
    
    msk_lbl = tkinter.Label(set_disp, text = "Mask Threshold:")
    msk_txt = tkinter.Text(set_disp, height = 1, width = 20)
    msk_txt.insert(tkinter.END, config.mask_thresh)
    msk_lbl.grid(row = 8, column = 0)
    msk_txt.grid(row = 8, column = 1)
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
        skiprow_chk = lkp_txt.get('1.0',tkinter.END).rstrip()
        try:
            value = int(skiprow_chk)
        except ValueError:
            tkinter.messagebox.showerror("Invalid Input", "Lineskips value must be integer")
            error_flag = True
        kL_file_chk = kl_txt.get('1.0',tkinter.END).rstrip()
        if (not kL_file_chk.endswith(".txt")):
            tkinter.messagebox.showerror("Invalid Input", "Left Camera Matrix file type must be .txt")
            error_flag = True
        elif(not os.path.isfile(mat_fold + kL_file_chk)):
            tkinter.messagebox.showerror("File Not Found", "Specified Left Camera Matrix file '" + mat_fold + kL_file_chk +
                                         "' not found.")
            error_flag = True
        kR_file_chk = kr_txt.get('1.0',tkinter.END).rstrip()
        if (not kR_file_chk.endswith(".txt")):
            tkinter.messagebox.showerror("Invalid Input", "Right Camera Matrix file type must be .txt")
            error_flag = True
        elif(not os.path.isfile(mat_fold + kR_file_chk)):
            tkinter.messagebox.showerror("File Not Found", "Specified Right Camera Matrix file '" + mat_fold + kR_file_chk +
                                         "' not found.")
            error_flag = True
        thresh_chk = thr_txt.get('1.0',tkinter.END).rstrip()
        try:
            value = float(thresh_chk)
        except ValueError:
            tkinter.messagebox.showerror("Invalid Input", "Correlation Threshold value must be float")
            error_flag = True
        msk_chk = msk_txt.get('1.0',tkinter.END).rstrip()
        try:
            value = int(msk_chk)
        except ValueError:
            tkinter.messagebox.showerror("Invalid Input", "Mask Threshold value must be integer")
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
            if(delim_txt.get('1.0',tkinter.END).rstrip() == ""):
                config.delim = " "
            else:
                config.delim = delim_txt.get('1.0',tkinter.END).rstrip()
            config.thresh = float(thr_txt.get('1.0',tkinter.END).rstrip())
            config.mask_thresh = int(msk_txt.get('1.0',tkinter.END).rstrip())
            set_disp.destroy()
    ok_btn = tkinter.Button(set_disp, text = "OK", command = ok_btn_click)
    
    cnc_btn.grid(row = 9, column = 0)
    ok_btn.grid(row = 9,column = 1)

set_btn = tkinter.Button(root, text = "Settings", command = set_window)
set_btn.grid(row = 3, column = 5)

root.mainloop()

    
