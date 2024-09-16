# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:19:52 2024

@author: myuey
"""
import tkinter as tk
from tkinter import filedialog
import confighandler as chand
import ncc_core as ncc
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import scripts as scr
import matplotlib.pyplot as plt
global config

version = 1.443


def start_recon(main_window):
    config = chand.ConfigHandler()
    config.load_config()
    recon_window = tk.Toplevel(main_window)
    recon_window.title("3D Stereo Reconstruction -MG- FSU Jena - v" + str(version))