# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:30:36 2023

@author: myuey
"""

from tkinter import Tk, Button, Entry, Label, messagebox, Toplevel, LEFT, W, filedialog
import numpy as np
import scripts as scr
import ncc_cor_core as ncc

def gui_initiate():
    root = Tk()
    root.title("3D Stereo Reconstruction - Matthew Guo - FSU Jena")
    root.geometry('900x400')
    root.resizable(width=False, height=False)
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file")
    root.mainloop()
gui_initiate()