# -*- coding: utf-8 -*-
"""
Create 2D figures from shape 3D cloud points.

@author: manon
"""


import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def plot_PCA(shape_file, path, figsize=(4,4)):
    """
    Draw shape_file points with a color 
    Project shape_file points onto the two main axes of the PCA. Then, save the
    plot in a .png file with the same name in the specified path.

    Parameters
    ----------
    shape_file : string
        File (with specified path) containing shape points.
    path : string
        Where to save the figure.
    figsize : tuple, optional
        Figure size in inches. The default is (5,5).

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


# Path of folder with folders containing the files from which to create images
path_raw = 'shapes_raw'

# Path of folder containing image folders
dataset_path = 'CNN_data'

for folder in glob.glob(path_raw+'/*'):
    
    folder_name = os.path.splitext(os.path.basename(folder))[0]
    dest = dataset_path + '/' + folder_name
    
    # Create images if they do not already exist
    if not os.path.exists(dest) :
        
        os.makedirs(dest) # creating destination folder
        
        # Browse all shape files (.txt ascii cloud data)
        for file in glob.glob(folder + '/*.txt'):
            plot_PCA(shape_file=file, path=dest)