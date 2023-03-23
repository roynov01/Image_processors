# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:52:32 2022

@author: royno
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
import os


root = tk.Tk()
batch = simpledialog.askinteger("batch number","what is the number to insert before image numbers?",parent=root)
directory = filedialog.askdirectory(title="choose directory with files to rename",parent=root)
# files = [f for f in os.listdir(directory) if os.path.isfile()]
root.destroy()


for file in os.listdir(directory):
    full_name = os.path.join(directory, file)
    print(f"**************\nold: {file}")
    if not os.path.isfile(full_name) or (not full_name.endswith(".tif") and not full_name.endswith(".ima")):
        continue
    name, num = file.split("(")
    num = num.split(")")[0]
    if len(num) == 1:
        add = str(batch) + "0" + num
    elif len(num) == 2:
        add = str(batch) + num
    elif len(num) == 3:
        add = str(batch) + num[1:]
    else:
        add = num
    
    if (file.endswith("MERGED.tif") or file.endswith("MERGED_MAX.tif") or "_M" in file):
        extension = "_MERGED.tif"
    elif file.endswith(".ima"):
        extension = ".ima"  
    else:
        extension = ".tif"
    
    new_name = f"{name}({add}){extension}"
    print(f"\nnew: {new_name}")
    full_name_new = os.path.join(directory, new_name)
    os.rename(full_name, full_name_new)
    

