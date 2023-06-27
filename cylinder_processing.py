# -*- coding: utf-8 -*-
"""
Program filtering detected cylinders (for example: cloud ASCII files exported
from CloudCompare) according to their orientation. Cylinders detected in each
plot are in the "cylinders_raw" folders and outputs are in the
"cylinders_filtered" folder.

@author: manon-col
"""


import os
import glob
import shutil
import numpy as np
import pandas as pd


class cylinder:
    """
    Manage a point cloud representing an (imperfect) cylinder model contained
    in a .txt file (cloud ASCII).
    
    """
    
    def __init__(self, file):
        
        # File containing cylinder data
        self._file = file
        # Filename without extension
        self._filename = os.path.splitext(os.path.basename(self._file))[0]
        # Reading file
        self._data = pd.read_csv(self._file, delimiter=' ', header=0)
        # Extracting coordinates
        self._xyz = self._data[['//X', 'Y', 'Z']].values
        self._filtered = False
    
    def orientation_filter(self, angle=45):
        """
        Filter of cylinder based on its horizontal orientation.
        
        """
                
        normals = self._data[['Nx', 'Ny', 'Nz']].values
        
        # Calculating the mean vector of the normals
        mean_vector = np.mean(normals, axis=0)
        
        # Normalizing the orientation vector
        normalized_vector = mean_vector/np.linalg.norm(mean_vector)
    
        # Computing the dot product between the normalized vector and the vertical axis (0, 0, 1)
        dot_product = np.dot(normalized_vector, np.array([0, 0, 1]))
    
        # Computing the angle between the vectors in radians
        angle_rad = np.arccos(dot_product)
    
        # Converting the angle to degrees
        angle_deg = np.degrees(angle_rad)

        # Checking if the angle is within the range of -angle to +angle degrees
        if abs(angle_deg) > angle: self._filtered = True
    
    def save(self, directory):
        """
        Save a copy of the cylinder file in a specified directory.
        
        """
        
        if self._filtered == False:
            dest = directory+'/'+self._filename+'.txt'
            shutil.copy2(self._file, dest)

def rename_all(path):
    """
    Rename all files in the folder whose path is specified, to have a
    continuous sequence of numbers.

    """
    
    files = os.listdir(path)
    counter = 1
    
    for file in files:
        
        old_path = os.path.join(path, file)
        extension = os.path.splitext(file)[1]  # get the file extension
        
        new_name = os.path.splitext(os.path.basename(path))[0] + '_' + \
            str(counter) + extension
        
        new_path = os.path.join(path, new_name)

        # Rename the file
        os.rename(old_path, new_path)

        counter += 1


## Processing

path_raw = 'cylinders_raw'
path_filtered = 'cylinders_filtered'

# Browsing all directories in initial path
for directory in glob.glob(path_raw+'/*'):

    dir_name = os.path.splitext(os.path.basename(directory))[0]

    # Checking if processing is not already done
    if dir_name not in os.listdir(path_filtered):
        
        # Create destination directory
        dest = path_filtered+'/'+dir_name
        os.makedirs(dest)
        
        # Browsing all .txt files
        for file in glob.glob(directory+'/*.txt'):
            
            cyl = cylinder(file)
            cyl.orientation_filter()
            cyl.save(dest)
        
        # Give clean names to saved files
        rename_all(dest)