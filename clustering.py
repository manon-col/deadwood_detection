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
from sklearn.pipeline import make_pipeline

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
        # List of cluster objects
        self._clusters = []
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
        
        for label in self._unique_labels:
            cluster_points = self._data_xyz[self._labels == label]
            self._clusters.append(cluster(label, cluster_points))
    
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
            
            # Filling the field
            for index in range(len(self._data_xyz)):
                new_las.cluster[index] = self._labels[index]+1
            
            # To solve an issue with point format / file version
            new_las = laspy.convert(new_las, point_format_id=7)
            # Saving las file with clustered points
            new_las.write(self._filename+'_clusters.las')
        
        else: print("Please run the clustering method first.")
        
 # /!\ the cylinder fitting part is in progress, doesn't work
 
    def fit_in_cluster(self, cluster_points, num_cylinders):
        """
        Fit a given number of cylinders in a given cloud points.

        Parameters
        ----------
        cluster_points : ndarray
            Array of points belonging to the cluster.
        num_cylinders : int
            Number of cylinders to fit.

        Returns
        -------
        cylinder_models : list
            List containing the fitted cylinder models.

        """
        
        cylinder_models = [] # Initialisation
        
        # Apply PCA to find the orientation of the points
        pca = PCA(n_components=3)
        pca.fit(cluster_points)
        orientation = pca.components_[0]

        # Fit multiple cylinders using RANSAC
        for _ in range(num_cylinders):

            # Create RANSAC regressor for cylinder fitting
            ransac = RANSACRegressor()

            # Create pipeline with PCA and RANSAC
            pipeline = make_pipeline(PCA(n_components=3), ransac)

            # Fit the pipeline to the cluster points
            pipeline.fit(cluster_points)

            # Get the inlier indices from RANSAC
            inlier_indices = ransac.inlier_mask_

            # Extract the inlier points
            inlier_points = cluster_points[inlier_indices]

            # Fit the cylinder again using all inliers
            pipeline.fit(inlier_points, inlier_points[:, 2])

             # Store the fitted cylinder model
            cylinder_model = {
                'center': pipeline.named_steps['ransacregressor'].estimator_.center_,
                'direction': orientation,
                'radius': pipeline.named_steps['ransacregressor'].estimator_.radius_,
                'height': np.max(inlier_points[:, 2]) - np.min(inlier_points[:, 2])
            }
            cylinder_models.append(cylinder_model)
        
        return cylinder_models

    def quality_measure(self, cluster_points, cylinder_model):
        """
        Calculate the quality measure for the given cylinder model and cluster points.
        The quality measure is the average distance between the points and the cylinder surface.
    
        Parameters
        ----------
        cluster_points : ndarray
            Array of points belonging to the cluster.
        cylinder_model : dict
            Dictionary containing the cylinder model parameters.
            Expected keys: 'center', 'direction', 'radius', 'height'
    
        Returns
        -------
        float
            Quality measure for the cylinder model.
    
        """
    
        center = cylinder_model['center']
        direction = cylinder_model['direction']
        radius = cylinder_model['radius']
    
        # Calculate the distance between points and cylinder surface
        distances = []
        for point in cluster_points:
            # Project the point onto the cylinder axis
            projection = np.dot(point - center, direction) * direction + center
            # Calculate the distance between the point and the projected point on the axis
            distance = np.linalg.norm(point - projection) - radius
            distances.append(distance)
    
        # Calculate the average distance as the quality measure
        quality_measure = np.mean(distances)
    
        return quality_measure

    def cylinder_fitting(self):
        """
        Fit cylinders into each cluster. Multiple iterations allow to know how
        many cylinders must be fit.

        """
        
        if self._labels is not None:
            
            for cluster in self._clusters:
                
                # Getting cluster points
                cluster_points = cluster.get_points()
                                
                # Initialise variables
                best_num_cylinders = 1
                best_measure = float('inf')
                
                # Iterate over different numbers of cylinders
                for num_cylinders in range(1, 20):
                    
                    # Fit cylinders using RANSAC
                    cylinder_models = self.fit_in_cluster(cluster_points,
                                                          num_cylinders)
                    
                    # Calculate quality measure for the cylinder fitting
                    measure = self.quality_measure(cluster_points,
                                                   cylinder_models)
                    
                    # Update the best number of cylinders if the measure improves
                    if measure < best_measure:
                        best_num_cylinders = num_cylinders
                    else:
                        # Stop iterating if the measure starts to degrade
                        break
                
                cylinder_models = self.fit_in_clusters(cluster_points,
                                                       best_num_cylinders)
                
                cluster.set_cylinders(cylinder_models)
                
        else: print("Please run the clustering method first.")

class cluster:
    """
    Manages a cluster of 3D points.
    
    """
    
    def __init__(self, label, points):
        self._label = str(label)
        self._points = points
        self._cylinder_models = None
    
    def label(self):
        return(self._label)
    
    def get_points(self):
        return(self._points)
    
    def set_cylinders(self, cylinder_models):
        """
        Store cylinders fitted to the cluster with the RANSAC method.

        Parameters
        ----------
        cylinder_models : list
            List containing all fitted cylinders.
        
        """
        
        self._cylinder_models = cylinder_models


# Test
test = deadwood_detection("output_computree.las")
test.clustering()
test.cylinder_fitting()

# notes...
# df= pd.DataFrame(columns=['cluster', 'orientation'])
# for cluster in test._unique_labels:
#     if cluster != -1:
#         cluster_points = test._data_xyz[test._labels == cluster]
#         pca=PCA(n_components=1)
#         pca.fit(cluster_points)
#         orientation = pca.components_[0]
#         # Normalize the orientation vector
#         normalized_orientation = orientation / np.linalg.norm(orientation)
    
#         # Define the vertical vector
#         vertical_vector = np.array([0, 0, 1])
    
#         # Calculate the dot product between the orientation and vertical vectors
#         dot_product = np.dot(normalized_orientation, vertical_vector)
    
#         # Calculate the angle using arccos
#         angle = np.degrees(np.arccos(dot_product))
#         df.loc[len(df)]=[int(cluster),angle]