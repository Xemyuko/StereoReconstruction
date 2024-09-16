# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:58:58 2024

@author: myuey
"""

import tkinter as tk
import recon_window as rw





root = tk.Tk()
root.title("Stereo Reconstruction Main Window")
def open_recon_window():
    rw.start_recon(root)
def open_cap_window():
    pass

rec_btn = tk.Button(root, text = "Reconstruction", command  = open_recon_window)
rec_btn.pack()
cap_btn = tk.Button(root, text = "Image Capture", command = open_cap_window)
cap_btn.pack()
root.geometry('100x100')
root.resizable(width=False, height=False)
root.focus_force()
root.mainloop()