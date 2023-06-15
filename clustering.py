# -*- coding: utf-8 -*-

import os
import laspy
import numpy as np
import random
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.linear_model import RANSACRegressor
# from sklearn.pipeline import make_pipeline

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
        # List of clusters
        self._clusters = []
        # Filename without extension
        self._filename = os.path.splitext(os.path.basename(self._las_file))[0]
    
    def clustering(self, eps=0.02, min_samples=5):
        """
        An euclidian clustering operation of points of the las file, using the
        DBSCAN algorithm.
        
        Parameters
        ----------
        eps : float
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other. This is not a maximum bound on
            the distances of points within a cluster. This is the most
            important DBSCAN parameter to choose appropriately for your data
            set and distance function. The default is 0.02.
        min_samples : float, optional
            The number of samples (or total weight) in a neighborhood for a
            point to be considered as a core point. This includes the point
            itself. The default is 100.
    
        """
                
        # Running DBSCAN algorithm
        clustering = DBSCAN(eps=0.05, min_samples=50).fit(self._data_xyz)
        
        # Clustering results, re-labelled to start from 0 instead of -1
        self._labels = clustering.labels_ + 1
    
        # Number of clusters in labels, ignoring noise (0) if present
        unique_labels = set(self._labels)
        n_clusters = len(unique_labels) - (1 if 0 in self._labels else 0)
        n_noise = list(self._labels).count(0)
        
        for label in unique_labels:
            
            # Getting current cluster points
            cluster_points = self._data_xyz[self._labels == label]
            
            # Creating cluster object
            self._clusters.append(cluster(label, cluster_points))
    
        print("Estimated number of clusters: %d" % n_clusters)
        print("Estimated number of noise points: %d" % n_noise)
    
    def draw_clusters(self):
        """
        Draw the clustering results, with each cluster in a different color and
        outliers in black.
        """
        
        if self._clusters:
            
            # Plotting clustering results
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            for cluster in self._clusters:
                
                if not cluster.is_filtered:
                    
                    if cluster.get_label() == 0:
                        # Outliers (noise) in black
                        color = 'k'
                    
                    else:
                        # Cluster points of the same (randomly chosen) color
                        color = '#'+''.join(random.choices('0123456789ABCDEF',
                                                           k=6))
                    
                    # Get current cluster points
                    cluster_points = cluster.get_points()
                    
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
        containing the cluster reference (noise is in "0" category).

        """
        if self._clusters:

            # Creating a copy of the original file
            new_las = self._las
            
            # Adding a new field "cluster"
            new_las.add_extra_dim(laspy.ExtraBytesParams(name="cluster",
                                                         type=np.uint8))
            
            # Filling the field
            for index in range(len(self._data_xyz)):
                
                label = self._labels[index]
                if not self._clusters[label].is_filtered:
                    new_las.cluster[index] = self._labels[index]
            
            # To solve an issue with point format / file version
            new_las = laspy.convert(new_las, point_format_id=7)
            
            # Saving las file with clustered points
            new_las.write(self._filename+'_clusters.las')
        
        else: print("Please run the clustering method first.")
        
    def filtering(self, nb_points=500):
        
        if self._clusters:
            
            n=0
            for cluster in self._clusters:
                if len(cluster.get_points()) < nb_points:
                    cluster.is_filtered = True
                    n+=1
            print(n+" clusters filtered.")
        
        else: print("Please run the clustering method first.")
    
    def reset_filtering(self):
        
        for cluster in self._clusters:
            cluster.is_filtered = False

class cluster:
    """
    Manage a cluster of 3D points.
    
    """
    
    def __init__(self, label, points):
        self._label = str(label)
        self._points = points
        self._cylinder_models = None
        self._filtered = False
    
    def get_label(self):
        return(self._label)
    
    def get_points(self):
        return(self._points)
    
    def is_filtered(self, boolean=None):
        """
        Update the filtering status if boolean (True or False) is specified.
        Else, return the filtering status. A filtered cluster will be removed
        from the point cloud.
        """
        
        if boolean == None: return(self._filtered)
        
        else: self._filtered = boolean

# Test
test = deadwood_detection("output_computree.las")
test.clustering()
test.filtering()
test.draw_clusters()
# test.save_clusters()
