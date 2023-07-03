# -*- coding: utf-8 -*-
"""
@author: manon-col
"""


import os
import laspy
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


class ClEngine:
    """
    Performs clustering on 3D point clouds obtained by LiDAR scanning to detect
    deadwood elements on the forest ground.

    This class focuses on the clustering step and assumes that pre-processing,
    including filtering and cleaning of ground and tall vegetation points, has
    already been performed.

    Methods:
    - __init__(self, las_file): Initialize the object and read the LAS file.
    - DBSCAN_clustering(self, eps=0.05, min_samples=100): Perform euclidean
    clustering using the DBSCAN algorithm.
    - draw_clusters(self): Draw the clustering results.
    - save_clusters(self): Save the clustering results in a new LAS file.
    - filtering(self, nb_points=500, min_dist=1): Filter clusters based on a
    minimum number of points and a minimum maximal length.
    - reset_filtering(self): Reset the filtering status of all clusters.
    
    """

    def __init__(self, las_file):
        """
        Initialize the object and read the LAS file.
        
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
        
        print("File "+self._filename+".las loaded successfully.")

    def DBSCAN_clustering(self, eps=0.05, min_samples=100):
        """
        An euclidian clustering operation of points of the las file, using the
        DBSCAN algorithm.
        
        Parameters
        ----------
        eps : float, optional
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other. This is not a maximum bound on
            the distances of points within a cluster. This is the most
            important DBSCAN parameter to choose appropriately for your data
            set and distance function. The default is 0.05m.
        min_samples : float, optional
            The number of samples (or total weight) in a neighborhood for a
            point to be considered as a core point. This includes the point
            itself. The default is 100.
    
        """
          
        print("Clustering "+self._filename+".las points...")
        
        # Running DBSCAN algorithm
        clustering=DBSCAN(eps=eps, min_samples=min_samples).fit(self._data_xyz)
        
        # Clustering results, re-labelled to start from 0 instead of -1
        self._labels = clustering.labels_+1
        
        # Number of clusters in labels, ignoring noise (0) if present
        unique_labels = set(self._labels)
        n_clusters = len(unique_labels) - (1 if 0 in self._labels else 0)
        n_noise = list(self._labels).count(0)
        
        for label in unique_labels:
            
            # Getting current cluster points
            cluster_points = self._data_xyz[self._labels == label]
            
            # Creating cluster object
            self._clusters.append(Cluster(label, cluster_points))
        
        print("Estimated number of clusters: %d" % n_clusters)
        print("Estimated number of noise points: %d" % n_noise)
    
    def draw_clusters(self):
        """
        Draw the clustering results, with points in the same cluster of the
        same colour.
        
        """
        
        print("Drawing clusters...")
        
        if self._clusters:
            
            # Plotting clustering results
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            for cluster in self._clusters:
                
                if not (cluster.is_filtered()):
                    
                    # Cluster points of the same (randomly chosen) color
                    random.seed(cluster.get_label())
                    color = '#' + ''.join(random.choices('0123456789ABCDEF',
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
            
            print("See plot.")
        
        else: print("Please run the clustering method first.")
    
    def save_clusters(self):
        """
        Save the clustering results in a new las file, with the cluster label
        in the classification field (noise is in "0" category).

        """
        
        print("Saving clustering results...")
        
        if self._clusters:
            
            # Creating a copy of the original file
            new_las = self._las
                        
            # Filling the "classification" field
            for index in range(len(self._data_xyz)):
                
                label = self._labels[index]
                
                if not self._clusters[label].is_filtered():
                    new_las.classification[index] = label
                                
            # To solve an issue with point format / file version
            new_las = laspy.convert(new_las, point_format_id=7)
            
            # Saving las file with clustered points
            new_las.write('cluster_outputs/'+self._filename+'_clusters.las')
            
            print("Clustering results successfully saved in "+self._filename+
                  "_clusters.las")
        
        else: print("Please run the clustering method first.")
        
    def filtering(self, nb_points=500, min_dist=1):
        """
        Basic cluster filter based on a minimum number of points and a minimum
        maximal length.

        Parameters
        ----------
        nb_points : integer, optional
            Minimum number of points a cluster must contain. The default is 500.
        min_dist: integer, optional
            Minimum distance that the 2 furthest points of the cluster must be
            from each other. The default is 1m.
            
        """
        
        print("Filtering clusters...")
        
        if self._clusters:
            
            n=0 # filtered cluster counter
            
            for cluster in self._clusters:
                
                # Filtering noise cluster
                if cluster.get_label() == 0:
                    
                    cluster.is_filtered(True)
                    n+=1
                
                # Filtering clusters that have above nb_points
                elif len(cluster.get_points()) < nb_points:
                    
                    cluster.is_filtered(True)
                    n+=1
                
                # Filtering clusters that have a length < min_dist
                else:
                                        
                    try:
                        
                        # Getting cluster points
                        points = cluster.get_points()
                        
                        # Calculating distances between all points
                        distances = cdist(points, points, 'euclidean')
                        
                        # Getting max distance
                        max_dist = np.max(distances)
                        
                        if max_dist < min_dist:
                            
                            cluster.is_filtered(True)
                            n+=1
                        
                    except MemoryError:
                        
                        print("MemoryError: Skipping cluster due to excessive"+
                              " memory usage.")
        
            print(str(n)+" clusters filtered.")
        
        else: print("Please run the clustering method first.")
    
    def reset_filtering(self):
        """
        Set the status of all clusters to unfiltered.
        
        """
        
        for cluster in self._clusters:
            cluster.is_filtered(False)


class Cluster:
    """
    Manage a cluster of 3D points.
    
    """
    
    def __init__(self, label, points):
        self._label = int(label)
        self._points = points
        self._filtered = False
    
    def __str__(self):
        return(f"Cluster {self._label}, {len(self._points)} points, filtering "
               +f"state: {self._filtered}")
    
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