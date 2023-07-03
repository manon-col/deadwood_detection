# -*- coding: utf-8 -*-
"""
Create images from 3D point clouds for subsequent classification.

@author: manon-col
"""


import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def save_plot(shape_file, path, figsize=(4,4)):
    """
    Plot shape_file points with a color depending on the z-coordinate. Then,
    save the plot in a .png file with the same name in the specified path.

    Parameters
    ----------
    shape_file : string
        File (with specified path) containing shape points.
    path : string
        Where to save the figure.
    figsize : tuple, optional
        Figure size in inches. The default is (4,4).

    """
    
    # Filename without extension
    filename = os.path.splitext(os.path.basename(shape_file))[0]
    
    # Read data and extract x,y,z coordinates
    data = pd.read_csv(shape_file, delimiter=' ', header=0)
    coords = data[['//X', 'Y', 'Z']].values
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a color gradient based on z-values
    cmap = plt.get_cmap('viridis')
    normalize = plt.Normalize(vmin=min(z), vmax=max(z))
    colors = cmap(normalize(z))
    
    # Generate graph
    ax.scatter(x, y,z, c=colors)
    
    # Hide axes
    ax.set_axis_off()
    
    # Save the figure as an image
    plt.savefig(path + '/' + filename + '.png', bbox_inches='tight',
                pad_inches=0)

    plt.close()


def make_images(data_folder, img_folder):
    """
    Create images from .txt files (stored in data_folder) in img_folder.
    
    """
    # Create images if they do not already exist
    if not os.path.exists(img_folder) :
    
        os.makedirs(img_folder) # creating destination folder
           
        # Browse all shape files (.txt ascii cloud data)
        for file in glob.glob(data_folder + '/*.txt'):
            save_plot(shape_file=file, path=img_folder)