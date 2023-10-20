# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:08:21 2023

@author: Admin
"""

#ml testscripts
import torch

def torch_test():
    print(torch.cuda.is_available())

    print(torch.cuda.device_count())


    print(torch.cuda.current_device())


    print(torch.cuda.device(0))


    print(torch.cuda.get_device_name(0))
    
torch_test()