


# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:27:49 2022

@author: Gregor_Gentsch
"""

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image  
import os
import numpy as np
import cv2
from vimba import *
import time
import sys
from telemetrix import telemetrix

#establish board and pins
dir_pin = 2 
pul_pin = 3

# # Create a Telemetrix instance.
board = telemetrix.Telemetrix()

# Set the DIGITAL_PIN as an output pin
board.set_pin_mode_digital_output(dir_pin)
board.set_pin_mode_digital_output(pul_pin)

#load in images
pattern_path = "C:\\Users\\IAOB\Documents\\tinkering\\BBM_Adrian\\1000p2p20\\"
pattern_file_names = os.listdir(pattern_path)
#comment next line out if using all patterns
pattern_file_names = pattern_file_names[:100]
pattern_file_list = []
for pattern_name in pattern_file_names:
    pattern = Image.open(pattern_path+pattern_name)
    pattern_file_list.append(pattern)

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

# turns the turntable
def turntable(steps_per_angle):
    for steps in range(steps_per_angle):
        board.digital_write(pul_pin, 1)
        time.sleep(0.005)
        board.digital_write(pul_pin, 0)
        time.sleep(0.005)
    return

            

#%%

mainwin = tk.Tk()
mainwin.title("Control window")
mainwin.geometry("600x400")


#Entry Window and Label for exposure time
expo_time_value = tk.IntVar(value=200000)
expo_time_label = ttk.Label(mainwin, text="Exposure time in µs", padding=[10])
expo_time_label.grid(column=0, row=1)
expo_time_entry = ttk.Entry(mainwin, text=expo_time_value)
expo_time_entry.grid(column=1, row=1)


#Entry Window and label for position start
pos_start_value = tk.IntVar(value=0)
pos_start_label = ttk.Label(mainwin, text="Starting Position", padding=[10])
pos_start_label.grid(column=2, row=1)
pos_start_entry = ttk.Entry(mainwin, text=pos_start_value)
pos_start_entry.grid(column=3, row=1)

#Stepper motor Steps, Microstepping and such
steps_per_rev = tk.IntVar(value=800)
steps_per_rev_label = ttk.Label(mainwin, text="Steps per revolution small motor", padding=[10])
steps_per_rev_label.grid(column=0, row=2)
steps_per_rev_entry = ttk.Entry(mainwin, text=steps_per_rev)
steps_per_rev_entry.grid(column=1, row=2)


#gear ratio, how big is the gear on the stepper motor compared to the big gear of the table
gear_ratio = tk.IntVar(value=12)
gear_ratio_label = ttk.Label(mainwin, text="Gear ratio", padding=[10])
gear_ratio_label.grid(column=0, row=3)
gear_ratio_entry = ttk.Entry(mainwin, text=gear_ratio)
gear_ratio_entry.grid(column=1, row=3)

#button to show minimal angle
def min_angle_compute():
    spr = steps_per_rev.get()
    gr = gear_ratio.get()
    min_angle = 360/(spr*gr)
    min_angle_string = str(min_angle) + " °"
    min_angle_label.configure(text=min_angle_string)
    
min_angle_button = ttk.Button(mainwin, text="Compute minimal angle")
min_angle_button.grid(column=0, row=4, sticky="ew")
min_angle_button.configure(command=min_angle_compute)

min_angle_label = ttk.Label(mainwin)
min_angle_label.grid(column=1, row=4)

#desired angle input, beware that is has to be a multiple of the minimal angle
des_angle = tk.DoubleVar(value=180)
des_angle_label = ttk.Label(mainwin, text="Desired angle per step", padding=[10])
des_angle_label.grid(column=0, row=5)
des_angle_entry = ttk.Entry(mainwin, text=des_angle)
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

steps_per_angle_button = ttk.Button(mainwin, text="Compute steps per desired angle", padding=[10])
steps_per_angle_button.grid(column=0, row=6, sticky="ew")
steps_per_angle_button.configure(command=steps_per_angle)

steps_per_angle_label = ttk.Label(mainwin, padding=[10])
steps_per_angle_label.grid(column=1, row=6)
    

#button to summon the projection window
def summ_proj_win():
    global proj_win
    proj_win = tk.Toplevel()
    proj_win.title("Projection Window")
    proj_win.geometry("600x600")
    img = ImageTk.PhotoImage(Image.open("C:\\Users\\IAOB\\Documents\\tinkering\\crosshair_small.png"))
    global proj_img_label
    proj_img_label = ttk.Label(proj_win, image=img)
    proj_img_label.image = img
    proj_img_label.place(relx=0.5, rely=0.5, anchor='center')
    
    

proj_win_button = ttk.Button(mainwin, text="Summon Projection Window", padding=[10])
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
        
    
    


start_projection_button = ttk.Button(mainwin, text="Start Projection", padding=[10])
start_projection_button.grid(column=0, row=8, sticky="ew")
start_projection_button.configure(command=projection_loop)






























mainwin.mainloop()

FSU-Cloud – ein sicherer Ort für all deine Daten
Impressum · Datenschutzerklärung


