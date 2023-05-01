# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:30:36 2023

@author: myuey
"""

import tkinter as tk
import confighandler as chand
import ncc_cor_core as ncc
confighand = chand.ConfigHandler()
#create window
root = tk.Tk()
root.title("3D Stereo Reconstruction - Matthew Guo - FSU Jena")
root.geometry('600x400')
root.resizable(width=False, height=False)
lang_opt = tk.StringVar(root)
options = ["English", "Deutsch"]
lang_opt.set(options[0]) 
'''
root.columnconfigure(2, minsize=10)
root.rowconfigure(0, minsize=50)
'''
#output filebox
out_label = tk.Label(root, text = "Output File:")
out_label.grid(row=0, column=0)
out_textbox = tk.Text(root, height=1, width=20)
out_textbox.grid(row=0, column=2)
#matrix folder location

def mat_btn_click():
    folder_path = tk.filedialog.askdirectory()
#images_L location
#images_R location
#interpolation points input
#offset value input
#start button

#save all fields as default button

#Help window
def inst_window():
    inst_disp = tk.Toplevel(root)
    inst_disp.title("Help")
    inst_disp.geometry('500x150')
    inst_disp.focus_force()
    inst_disp.resizable(width=False, height=False)

    
#instructions button
inst_btn = tk.Button(root, text = "Help", command = inst_window)
#advanced settings window
def adv_set_window():
    inst_disp = tk.Toplevel(root)
    if lang_opt.get() == "English":
        inst_disp.title("Settings")
    else:
        inst_disp.title("Einstellungen")
    inst_disp.geometry('400x600')
    inst_disp.focus_force()
    inst_disp.resizable(width=False, height=False)
#language selector
lang_label = tk.Label(root, text = "Language:")
lang_label.grid(row=0, column=4)


set_btn = tk.Button(root, text = "Settings", command = adv_set_window)
set_btn.grid(row = 1, column = 4)
def update_text_language(lang_sel):
    if lang_sel == "English":
        out_label.config(text = "Output File:")
        lang_label.config(text = "Language:")
        set_btn.config(text = "Settings")
    else:
        out_label.config(text = "Datei Ausgaben:")
        lang_label.config(text = "Sprache:")
        set_btn.config(text = "Einstellungen")
dropdown = tk.OptionMenu(root, lang_opt, *options, command=update_text_language)
dropdown.grid(row=0, column=5)
#advanced settings button


root.mainloop()

    
