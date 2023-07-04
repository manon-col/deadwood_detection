# -*- coding: utf-8 -*-
"""
Simple program that converts all las files in the "to_convert" directory into
point format 7 (to avoid errors with CloudCompare for ex). All converted files
are saved in the "converted" directory. Files in the "converter" directory are
not supposed to stay here forever. Note that merger.py also converts las files.

@author: manon-col
"""

import laspy
import os
import glob

def convert(file):
    
    # Reading file
    las = laspy.read(file)
    
    # File name without extension
    filename = os.path.splitext(os.path.basename(file))[0]
    
    # Creating new las file
    new_las = las
    
    # Conversion
    new_las = laspy.convert(new_las, point_format_id=7)
    
    # Saving converted file
    new_las.write('converted/' + filename + '_pt7.las')

# Listing all las files
las_files = glob.glob('to_convert/*.las')

for file in las_files:
    convert(file)