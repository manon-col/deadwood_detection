# -*- coding: utf-8 -*-
"""
Simple program that merges all las files in the "to_merge" directory, and saves
the result in the "merged" directory. Files in the "merger" directory are not
supposed to stay here forever.

@author: manon-col
"""

import laspy
import glob

# Listing all las files
las_files = glob.glob('to_merge/*.las')

# Initialise new las file
path_out = 'merged/merged_file.las'
new_las = laspy.create(point_format=7)
new_las.write(path_out)

for file in las_files:
    
    # Read las file
    las_in = laspy.read(file)
    
    # Append points in new file
    with laspy.open(path_out, mode="a") as las_out:
        las_out.append_points(las_in.points)