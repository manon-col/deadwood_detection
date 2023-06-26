# -*- coding: utf-8 -*-
"""

@author: manon
"""


import os
import glob
import pandas as pd
import numpy as np


class cylinder:
    """
    Manage a point cloud representing an (imperfect) cylinder model contained
    in a ASCII file.
    
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
    
    def orientation_filter(self):
        """
        Filter of cylinder based on the
        
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

        # Checking if the angle is within the range of -45 to +45 degrees
        if abs(angle_deg) > 45: self._is_filtered = True
    

# Listing all raw cylinder files
txt_files = glob.glob('cylinders_raw/07_04/*.txt')

for file in txt_files:
    print(file)
    c=cylinder(file)
    c.orientation()