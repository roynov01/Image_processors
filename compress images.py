# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 13:15:53 2022

@author: royno
"""

from PIL import Image

# https://stackoverflow.com/questions/10607468/how-to-reduce-the-image-file-size-using-pil
PATH = r"X:\roy\apicome\visium_export_from_yotam"
sample_names = ["P1","P5","P8","P9","P10","P11","P12","P13"]

Image.MAX_IMAGE_PIXELS = 999999999*2

for i in range(len(sample_names)):
    img = Image.open(PATH + "\\" + sample_names[i] + ".png")
    print("processing: " + sample_names[i])
    img = img.resize((2500,2500), Image.ANTIALIAS)
    img.save(PATH + "\\" + sample_names[i] + "_compressed.png", quality=95)