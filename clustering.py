# -*- coding: utf-8 -*-

import os
import laspy
import numpy as np
import random
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor

class deadwood_detection:
    """
    Methods for processing 3D point clouds obtained by LiDAR scanning, in order
    to detect deadwood elements on the forest ground. Pre-processing must be
    already performed : 3D cloud points must be filtered, as well as cleaned
    from ground and tall vegetation points as far as possible.
    
    """

    def __init__(self, las_file):
        """
        Initiate reading of the las file, extracting the x,y,z coordinates of
        the points.
        
        Parameters
        ----------
        las_file : LasData file
            File containing the points to process.
        """
        # File
        self._las_file = las_file
        # File reading
        self._las = laspy.read(self._las_file)
        # Array of all points x, y, z coordinates
        self._data_xyz = self._las.xyz
        # Clustering results
        self._labels = None
        # Set of clustering labels
        self._unique_labels = None
        # Filename without extension
        self._filename = os.path.splitext(os.path.basename(self._las_file))[0]
    
    def clustering(self, eps=0.05, min_samples=100):
        """
        Clusters points of the las file, using the DBSCAN algorithm.
        
        Parameters
        ----------
        eps : float
            The maximum distance between two samples for one to be considered as in
            the neighborhood of the other. This is not a maximum bound on the
            distances of points within a cluster. This is the most important DBSCAN
            parameter to choose appropriately for your data set and distance
            function. The default is 0.05.
        min_samples : float, optional
            The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point. This includes the point itself. The
            default is 100.
    
        """
                
        # Running DBSCAN algorithm
        clustering = DBSCAN(eps=0.1, min_samples=50).fit(self._data_xyz)
        self._labels = clustering.labels_
    
        # Number of clusters in labels, ignoring noise if present
        self._unique_labels = set(self._labels)
        n_clusters = len(self._unique_labels) - (1 if -1 in self._labels else 0)
        n_noise = list(self._labels).count(-1)
    
        print("Estimated number of clusters: %d" % n_clusters)
        print("Estimated number of noise points: %d" % n_noise)
    
    def draw_clusters(self):
        """
        Draw the clustering results, with each cluster in a different color and
        outliers in black.
        """
        
        if self._labels is not None :
            
            # Plotting clustering results
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
            for label in self._unique_labels:
                if label == -1:
                    # Outliers (noise) in black
                    color = 'k'
                else:
                    # Cluster points of the same (randomly chosen) color
                    color = '#' + ''.join(random.choices('0123456789ABCDEF',
                                                         k=6))
            
                # Get current cluster points
                cluster_points = self._data_xyz[self._labels == label]
                
                # Draw cluster points
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                            cluster_points[:, 2], c=color, marker='o')
        
            # Axes and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Clustering results')
            
            # Showing the plot
            plt.show()
        
        else: print("Please run the clustering method first.")
    
    def save_clusters(self):
        """
        Save the clustering results in a new las file, with an extra dimension
        containing the cluster reference (noise is in "-1" category).

        """
        if self._labels is not None:

            # Creating a copy of the original file
            new_las = self._las
            
            # Adding a new field "cluster"
            new_las.add_extra_dim(laspy.ExtraBytesParams(name="cluster",
                                                         type=np.uint8))
            
            # Browsing all the data to find out which label goes to which point,
            # and  fill "clusters" field. Must be optimised...
            for index in range(len(self._data_xyz)):
                new_las.cluster[index] = self._labels[index]+1
            
            # To solve an issue with point format / file version
            new_las = laspy.convert(new_las, point_format_id=7)
            # Saving las file with clustered points
            new_las.write(self._filename+'_clusters.las')
        
        else: print("Please run the clustering method first.")
    
    