# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:30:36 2023

@author: myuey
"""

from tkinter import Tk, Text,Button, Entry, Label, messagebox, Toplevel, LEFT, W, filedialog


def gui_initiate():
    root = Tk()
    root.title("3D Stereo Reconstruction - Matthew Guo - FSU Jena")
    root.geometry('900x400')
    root.resizable(width=False, height=False)
    lab1 = Label(root, text = "Target")
    lab1.config(font =("Courier", 14))
    t1 = Text(root, height = 1, width = 20)
    b1 = Button(root, text = "Directory")
    lab1.pack()
    t1.pack()
    b1.pack()
    #root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file")
    root.mainloop()
gui_initiate()