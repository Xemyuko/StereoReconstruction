# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:58:58 2024

@author: myuey
"""

import tkinter as tk
import recon_window as rw



version = 0.1

root = tk.Tk()
root.title("Stereo Reconstruction FSU Jena - v" + str(version))
def open_recon_window():
    rw.start_recon(root)
def open_cap_window():
    pass
def open_cal_winodw():
    pass
rec_btn = tk.Button(root, text = "Reconstruction", command  = open_recon_window)
rec_btn.pack()
cap_btn = tk.Button(root, text = "Image Capture", command = open_cap_window)
cap_btn.pack()
root.geometry('400x200')
root.resizable(width=False, height=False)
root.focus_force()
root.mainloop()