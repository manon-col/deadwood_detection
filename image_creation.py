# -*- coding: utf-8 -*-
"""
Create images from 3D point clouds for subsequent classification.

@author: manon-col
"""


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_plot(shape_file, path, figsize, dpi):
    """
    Plot shape_file points with a color depending on the z-coordinate. Optimise
    the view by rotating the image on the z axis to allow the maximum number of
    points to be seen. Then, save the plot in a .png file with the same name in
    the specified path.

    Parameters
    ----------
    shape_file : string
        File (with specified path) containing shape points.
    path : string
        Where to save the figure.
    figsize : tuple, optional
        Figure size in inches.
    dpi : integer
        Dots per inch.

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
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a color gradient based on z-values
    cmap = plt.get_cmap('viridis')
    normalize = plt.Normalize(vmin=min(z), vmax=max(z))
    colors = cmap(normalize(z))
    
    # Generate graph
    ax.scatter(x, y,z, c=colors)
    
    # Hide axes
    ax.set_axis_off()
    
    # Calculate centroid of the point cloud
    centroid = np.mean(coords, axis=0)
    
    # Calculate vector from centroid to each point
    vectors = coords - centroid
    
    # Convert to polar coordinates
    azimuths = np.array(list(map(cartesian_to_spherical, vectors)))[:, 1]
    
    # Calculate average azimuth angle
    average_azimuth = np.mean(azimuths) * 180 / np.pi
    
    # Set the view angle based on average azimuth
    ax.view_init(azim=average_azimuth)
    
    # Save the figure as an image
    plt.savefig(path + '/' + filename + '.png', dpi=dpi)
    
    plt.close()


def cartesian_to_spherical(coordinates):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    coordinates : array-like
        Cartesian coordinates [x, y, z].

    Returns
    -------
    r : float
        Radial distance.
    theta : float
        Azimuth angle in radians.
    phi : float
        Polar angle in radians.
    """
    x, y, z = coordinates
    r = np.linalg.norm(coordinates)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    
    return r, theta, phi


def image_generator(data_folder, img_folder, figsize = (4,4), dpi=75):
    """
    Create images from multiple .txt files (containing points of a 3D shape,
    stored in data_folder) in img_folder.
    
    Parameters
    ----------
    data_folder : string
        Path of folder containing all shape files (.txt ascii cloud data).
    img_folder : string
        Where to save the figures.
    figsize : tuple, optional
        Figure size in inches. The default is (4,4)
    dpi : integer
        Dots per inch. The default is 75.
    """
    # Create destination folder if needed
    if not os.path.exists(img_folder): os.makedirs(img_folder)
    
    # Create images if they do not already exist
    if len(os.listdir(img_folder)) == 0:
        
        print(f"Creating images in {img_folder}")
        
        # Browse all shape files
        for file in glob.glob(data_folder + '/*.txt'):
            save_plot(shape_file=file, path=img_folder, figsize=figsize,
                      dpi=dpi)