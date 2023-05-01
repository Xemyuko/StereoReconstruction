# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:21:08 2023

@author: myuey
"""
import ncc_cor_core as ncc
import confighandler as chand
confighand = chand.ConfigHandler()
confighand.load_config()
ncc.run_cor(confighand)
