# -*- coding: utf-8 -*-
"""
Program for clustering points from a .las file, filtering, and managing
clustering outputs.

@author: manon-col
"""


import os
import csv
import time
import laspy
import random
import warnings
import alphashape
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN, validity
from sklearn.base import BaseEstimator
from scipy.spatial.distance import cdist
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, silhouette_score


class ClEngine:
    """
    Program for clustering points from a .las file, filtering, and managing
    clustering outputs.
    
    This class performs clustering on 3D point clouds obtained by LiDAR
    scanning to detect deadwood elements on the forest ground. It focuses on
    the clustering step and assumes that pre-processing, including filtering
    and cleaning of ground and tall vegetation points, has already been
    performed.
    
    Methods:
        
    - __init__(self, las_file): Initialize the object and read the LAS file.
    - DBSCAN_clustering(self, eps=0.05, min_samples=100): Perform euclidean
      clustering using the DBSCAN algorithm.
    - HDBSCAN_clustering(self, min_cluster_size=500,
      max_cluster_size=200000, min_samples=10): Cluster data finely using
      hierarchical density-based clustering.
    - get_clusters(self): Return the last created list of clusters.
    - filtering(self, nb_points=None, coord_file=None, sep=';', dec=',',
      distance_from_centre=18, delta=0.5, min_dist=1): Basic cluster filter
      based on various criteria.
    - keep_clusters(self, cluster_list): Keep only clusters whose index is in
      cluster_list, filter the others.
    - reset_filtering(self): Set the status of all clusters to unfiltered.
    - save_all_points(self, folder, suffix): Save the clustering results in a
      new LAS file, in the specified folder, with the cluster label in the
      'cluster' field (filtered clusters are not saved).
    - save_individual_points(self, folder): Create and save .las files from
      each cluster, in a sub-folder of the specified directory.
    - save_individual_images(self, folder, figsize=(4, 4), dpi=75): Create
      and save .png images from each cluster, in a sub-folder of the specified
      directory.
    - draw_clusters(self): Draw the clustering results (simply), with points
      in the same cluster of the same colour.
    
    """
    
    def __init__(self, las_file):
        """
        Initialize the object and read the LAS file.

        Parameters
        ----------
        las_file : LasData file
            File containing the points to process.

        """
        
        if not las_file.lower().endswith('.las'):
            raise ValueError("The file must have the .las extension.")

        # File
        self._las_file = las_file
        # File reading
        self._las = laspy.read(self._las_file)
        # Array of all points x, y, z coordinates
        self._data_xyz = self._las.xyz
        # Clustering results
        self._labels = None
        # List of raw clusters
        self._raw_clusters = []
        # List of clusters to save
        self._clusters = []
        # Filename without extension
        self._filename = os.path.splitext(os.path.basename(self._las_file))[0]
        
        # Load existing clusters
        if hasattr(self._las, 'cluster'):
            
            for cluster in range(1, len(np.unique(self._las.cluster))+1):
                
                # Load existing volumes
                if hasattr(self._las, 'volume'):
                    
                    self._raw_clusters.append(
                        Cluster(
                            label=cluster,
                            points=self._data_xyz[self._las.cluster==cluster],
                            volume=self._las.volume[
                                self._las.cluster==cluster][0]))
                else:
                    
                    self._raw_clusters.append(
                        Cluster(
                            label=cluster,
                            points=self._data_xyz[self._las.cluster==cluster]))
                    
            self._clusters = self._raw_clusters
        
        if len(self._clusters)==0:
            print(f"File {self._filename}.las loaded successfully.")
        
        else:
            print(f"File {self._filename}.las loaded successfully, "+
                  f"{len(self._clusters)} clusters found.")
        
    def DBSCAN_clustering(self, tuning=False, n_iter=10, display=False,
                          eps=0.05, min_samples=100):
        """
        An euclidean clustering operation of points of the las file, using the
        DBSCAN algorithm.

        Parameters
        ----------
        tuning : boolean, optional
            If True, DBSCAN hyperparameter tuning is performed on the epsilon
            and min_samples parameters. The default is False.
        n_iter : integer, optional
            The number of iterations for randomized search. The default is 10.
        display : boolean, optional
            If True, print the best parameters found during hyperparameter
            tuning. The default is false.
        eps : float, optional
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other. This is not a maximum bound on
            the distances of points within a cluster. This is the most
            important DBSCAN parameter to choose appropriately for your data
            set and distance function. Not taken into account if tuning is
            True. The default is 0.05m.
        min_samples : float, optional
            The number of samples (or total weight) in a neighborhood for a
            point to be considered as a core point. This includes the point
            itself. Not taken into account if tuning is True. The default is
            100.

        """
        
        print(f"Clustering {self._filename}.las points...")
        
        start = time.time()
        
        if tuning is True:
            
            # Run DBSCAN algorithm and perform hyperparameter tuning
            clustering_tuner = ClusteringTuner(cluster_type='dbscan',
                                               n_iter=n_iter)
            
            # Best clustering results
            self._labels = clustering_tuner.fit_predict(self._data_xyz)
            
            if display is True:
                best_params = clustering_tuner.get_best_params()
                print(f"Best Parameters: {best_params}")
        
        else:
            
            # Run DBSCAN algorithm
            clustering = DBSCAN(eps=eps,
                                min_samples=min_samples)
            clustering.fit(self._data_xyz)

            # Clustering results
            self._labels = clustering.labels_

        # Number of clusters in labels, ignore noise (0) if present
        unique_labels = set(self._labels)

        for label in unique_labels:
            
            # Ignore noise points
            if label != -1:
                
                # Get current cluster points
                cluster_points = self._data_xyz[self._labels == label]
    
                # Create cluster object
                self._raw_clusters.append(
                    Cluster(label+1, # 1st cluster = cluster 1 instead of 0
                            cluster_points))
        
        end = time.time()
        elapsed_time = self._timer(start, end)
        
        print("DBSCAN clustering done, estimated number of clusters: " +
              f"{len(self._raw_clusters)}, elapsed time: " + elapsed_time)
    
    def HDBSCAN_clustering(self, tuning = False,
                           n_iter=10,
                           n_test=None,
                           display=False,
                           min_cluster_size=500,
                           max_cluster_size=200000,
                           min_samples=10):
        """
        Cluster data finely using hierarchical density-based euclidean
        clustering. The data must already have been clustered by DBSCAN
        algorithm, as HDBSCAN is used on each cluster (otherwise the processing
        time is too long).
        
        Parameters
        ----------
        tuning : boolean, optional
            If True, DBSCAN hyperparameter tuning is performed on the
            min_samples and min_cluster_size parameters. The default is False.
        n_iter : integer, optional
            The number of iterations for randomized search. The default is 10.
        n_test : integer, optional
            When the best parameters found during hyperparameter tuning are the
            same (or almost) n_test times, these parameters are chosen for the
            following clusters in order to save time (only if not None). The
            default is None.
        display : boolean, optional
            If True, print the best parameters found during hyperparameter
            tuning. The default is False.
        min_cluster_size : integer, optional
            The minimum number of samples in a group for that group to be
            considered a cluster; groupings smaller than this size will be left
            as noise. Not taken into account if tuning is True. The default is
            500.
        max_cluster_size : integer, optional
            The maximum number of samples a cluster can have. Not taken into
            account if tuning is True. The default is 200000.
        min_samples : integer, optional
            The number of samples in a neighborhood for a point to be
            considered as a core point. This includes the point itself. Not
            taken into account if tuning is True.The default is 10.
        
        """
        
        if self._raw_clusters:
            
            cl_list = self._raw_clusters # copy of cluster list
            self._raw_clusters = [] # re-initialise
            
            param_dict= {'tuning': tuning,
                         'min_cluster_size': min_cluster_size,
                         'max_cluster_size': max_cluster_size,
                         'min_samples': min_samples
                         }
            
            best_params_list = []
            
            print("Performing HDBSCAN fine clustering...")
            
            start = time.time()
            
            for cluster in cl_list:
                
                # Get current cluster points
                points = cluster.get_points()
                
                # Avoid unnecessary clustering of small clusters
                if param_dict['tuning'] is False and len(cluster.get_points())\
                    >= param_dict['min_cluster_size']:
                    
                    # Run HDBSCAN algorithm
                    clustering = HDBSCAN(
                        min_cluster_size=param_dict['min_cluster_size'],
                        max_cluster_size=param_dict['max_cluster_size'],
                        min_samples=param_dict['min_samples'])
                    
                    clustering.fit(points)
                    labels = clustering.labels_
                
                elif param_dict['tuning'] is True:
                    
                    # Run HDBSCAN algorithm and perform hyperparameter tuning
                    clustering_tuner = ClusteringTuner(
                        cluster_type='hdbscan', n_iter=n_iter)
                    
                    # Best clustering results
                    labels = clustering_tuner.fit_predict(points)
                    
                    best_params = clustering_tuner.get_best_params()
                    best_params_list.append(best_params)
                    
                    if display is True:
                        print(f"Best Parameters: {best_params}")
                    
                    if n_test is not None and len(best_params_list) == n_test:
                        
                        keys = [(params['min_samples'],
                                 params['min_cluster_size'],
                                 params['max_cluster_size']) \
                                for params in best_params_list]
                        
                        occurrences = Counter(keys)
                        most_common = occurrences.most_common(1)[0][0]
                        
                        # Using best parameters to continue clustering, stop
                        # hyperparameter tuning
                        param_dict['tuning'] = False
                        param_dict['min_samples'] = most_common[0]
                        param_dict['min_cluster_size'] = most_common[1]
                        param_dict['max_cluster_size'] = most_common[2]
                        
                        print(f"Using min_cluster_size={most_common[1]}, "+
                              f"max_cluster_size={most_common[2]}, "+
                              f"min_samples={most_common[0]} to continue "+
                              "hdbscan clustering as they were the best\n"+
                              f"parameters {occurrences.most_common(1)[0][1]}"+
                              f" times out of {n_test}...")
                
                else: labels = None
                
                if labels is not None:
                    
                    unique_labels = set(labels)
                    
                    # If no subscluster found (only noise and eventually 1
                    # cluster found with hdbscan), keep dbscan cluster
                    if len(unique_labels) <= 2:
                        
                        new_cluster_points = points
                        
                        # Create cluster object
                        self._raw_clusters.append(
                            Cluster(label=len(self._raw_clusters)+1,
                                    points=new_cluster_points))
                    
                    else:
                        
                        for label in unique_labels:
                            
                            # Ignore noise points
                            if label != -1:
                                
                                # Get current cluster points
                                new_cluster_points = points[labels == label]
                                
                                # Create cluster object
                                self._raw_clusters.append(
                                    Cluster(label=len(self._raw_clusters)+1,
                                            points=new_cluster_points))
            
            end = time.time()
            elapsed_time = self._timer(start, end)  
            
            print("HDBSCAN clustering done, estimated number of clusters: " +
                  f"{len(self._raw_clusters)}, elapsed time: " + elapsed_time)
        
        else: print("Please run DBSCAN_clustering first.")
        
    def get_clusters(self):
        """
        Return the last created list of clusters.
        
        """
        
        if self._clusters: return self._clusters
        elif self._raw_clusters: return self._raw_clusters
        else: print("No cluster found.")
    
    def filtering(self,
                  nb_points=None,
                  delta=None,
                  min_dist=None,
                  coord_file=None,
                  distance_from_centre=None,
                  sep=';',
                  dec=','):
        """
        Basic cluster filter based on a minimum number of points, a maximum
        distance from plot centre, a maximum distance from the ground, and a
        minimum length.

        Parameters
        ----------

        nb_points : integer, optional
            Minimum number of points a cluster must contain. Set to None to
            ignore point density filtering. The default is None.
        delta : float, optional
            Maximum distance from the ground. Set to None to ignore height
            filtering. The default is None.    
        min_dist: integer, optional
            Minimum distance that the 2 furthest points of the cluster must be
            from each other. Set to None to ignore length filtering. The
            default is 1m.    
        coord_file : string, optional
            Path leading to the csv file that contains coordinates of the plot
            centre. If set to None, the centre coordinates are 0,0. The
            default is None.
        distance_from_centre : integer, optional
            Actual radius of the inventory plot. Set to None to ignore
            distance_from_centre filtering. The default is None.
        sep : string, optional
            Seperator of the csv file. The default is ';'.
        dec : string, optional
            Decimal separator of the csv file. The default is ','.

        """
        
        print("Filtering clusters...")

        if self._raw_clusters:
            
            # with np.errstate(invalid='ignore'):
                
            for cluster in self._raw_clusters:

                cluster.nb_points_filter(nb_points=nb_points)
                cluster.distance_from_centre_filter(
                    plot_name=self._filename,
                    distance=distance_from_centre,
                    coord_file=coord_file,
                    sep=sep,
                    dec=dec)
                cluster.flying_filter(delta=delta)
                cluster.length_filter(min_dist=min_dist)

            self._create_clusters_list()

            print(f"{len(self._clusters)} clusters remaining out of " +
                  f"{len(self._raw_clusters)}.")

        else: print("Please run the clustering method first.")
    
    def keep_clusters(self, cluster_list, relabel=True):
        """
        Keep only clusters whose index is in cluster_list, filter the others.

        Parameters
        ----------
        cluster_list : list
            List of clusters (cluster indexes/labels) to keep.
        relabel : boolean
            If True, the clusters are relabelled starting from 1. The default
            is True.
            
        """
        
        if cluster_list:
            
            cluster_list = [int(label) for label in cluster_list]
            
            for cluster in self._raw_clusters:
                
                if cluster.get_label() in cluster_list:
                    cluster.is_filtered(False)
                    
                else: cluster.is_filtered(True)
            
            self._create_clusters_list(relabel=relabel)
    
    def reset_filtering(self):
        """
        Set the status of all clusters to unfiltered.

        """

        for cluster in self._raw_clusters: cluster.is_filtered(False)
        
        self._create_clusters_list()

    def _create_clusters_list(self, relabel=True):
        """
        Create the list of clusters to save from unfiltered clusters.
        
        Parameters
        ----------
        relabel : boolean
            If True, the clusters are relabelled starting from 1. The default
            is True.
            
        """
        
        # Empty list
        self._clusters = []
        
        for cluster in self._raw_clusters:
            
            if not cluster.is_filtered(): self._clusters.append(cluster)

        if relabel is True:
            
            # Relabel clusters
            for index in range(len(self._clusters)):
                
                self._clusters[index].set_label(index+1)

    def save_all_points(self, folder, suffix=''):
        """
        Save the clustering results in a new las file, in the specified folder,
        with the cluster label in the 'cluster' field (filtered clusters are
        not saved).

        Parameters
        ----------
        folder : string
            Where to save the file.
        suffix : string, optional
            Filename suffix before extension. The default is ''.

        """

        if not os.path.exists(folder): os.makedirs(folder)
        
        if self._raw_clusters or self._clusters:

            # Create new .las file
            path_out = f'{folder}/{self._filename}_{suffix}.las'
            new_las = laspy.create(point_format=7, file_version="1.4")
            
            # Add a new field "cluster"
            new_las.add_extra_dim(laspy.ExtraBytesParams(name="cluster",
                                                         type=np.uint32))
            
            # Add "volume" field if calculated
            if self._clusters and self._clusters[0].get_volume() is not None:
                new_las.add_extra_dim(laspy.ExtraBytesParams(name="volume",
                                                             type=np.float64))
            new_las.header.scales = np.array([1.e-05, 1.e-05, 1.e-05])
            new_las.write(path_out)

            # Filling the "cluster" field
            
            if not self._clusters:
                
                for cluster in self._raw_clusters:

                    points = cluster.las_points(header=new_las.header)

                    # Append .las points to new file
                    with laspy.open(path_out, mode="a") as las_out:
                        las_out.append_points(points)
                
                print(f"{len(self._raw_clusters)} clusters successfully saved"+
                      f" in {path_out}.")
            
            else:
                
                print("Saving unfiltered clusters in .las file...")
                
                for cluster in self._clusters:
                    
                    points = cluster.las_points(header=new_las.header)
    
                    # Append .las points to new file
                    with laspy.open(path_out, mode="a") as las_out:
                        las_out.append_points(points)
    
                print(f"{len(self._clusters)} clusters successfully saved in "+
                      f"{path_out}.")

        else: print("Please run the clustering method first.")
    
    def save_individual_points(self, folder, suffix=''):
        """
        Create and save .las files from each cluster, in a sub-folder of the
        specified directory.

        Parameters
        ----------
        folder : string
            Parent folder for cluster files.
        suffix : string, optional
            Las file name suffix before extension. The default is ''.

        """
        
        if self._clusters:

            print("Saving clusters in individual .las files...")
            save_path = f'{folder}/{self._filename}'
            
            for cluster in self._clusters:
                cluster.create_las(save_path=save_path, prefix=self._filename,
                                   suffix=suffix)
            
            print(f"Point clouds successfully saved in {save_path}.")

        else: print("Please filter clusters first.")
    
    def save_individual_images(self, folder, figsize=(4, 4), dpi=75):
        """
        Create and save .png images from each cluster, in a sub-folder of the
        specified directory.

        Parameters
        ----------
        folder : string
            Parent folder for cluster images.
        figsize : tuple, optional
            Figure size in inches. The default is (4,4).
        dpi : integer, optional
            Dots per inch. The default is 75.

        """

        if self._clusters:

            print("Saving clusters in individual .png files...")

            for cluster in self._clusters:

                save_path = f'{folder}/{self._filename}'

                # Create destination folder if needed
                if not os.path.exists(save_path): os.makedirs(save_path)

                cluster.create_img(save_path=save_path,
                                   prefix=self._filename,
                                   figsize=figsize, dpi=dpi)

            print(f"Images successfully saved in {save_path}.")

        else: print("Please filter clusters first.")
    
    def _timer(self, start, end):
        """
        Give elapsed time between start and end (float time objects).
        
        """
        
        elapsed_seconds = int(end - start)
    
        hours = elapsed_seconds // 3600
        minutes = (elapsed_seconds % 3600) // 60
        seconds = elapsed_seconds % 60
    
        return f"{hours}h{minutes}min{seconds}s"
    
    def draw_clusters(self):
        """
        Draw the clustering results (simply), with points in the same cluster
        of the same colour.

        """

        print("Drawing clusters...")

        if self._clusters: to_draw = self._clusters
        
        elif self._raw_clusters: to_draw = self._raw_clusters
        
        else:
            print("Please run the clustering method first.")
            return

        # Plotting clustering results
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for cluster in to_draw:

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
    
    def calculate_volumes(self, alpha=50, plot=False):
        """
        For each cluster, create an alpha shape (concave hull) from the cluster
        points and calculate its volume.

        Parameters
        ----------
        alpha: integer, optional
            Alpha parameter for alpha shape creation. The higher the alpha
            value is, the tighter the bounding shape will fit the data. The
            default is 50.
        plot: boolean, optional
            If True, a plot of the alpha shape is shown. The default is False.
            
        """
        
        if self._clusters:
            
            print("Calculating cluster volumes...")
            
            # Calculate volume of all clusters
            for cluster in self._clusters:
                cluster.calculate_volume(alpha=alpha, plot=plot)
                
        else: print("Please run the clustering method first.")
        
    def save_volumes_csv(self, folder, suffix=''):
        """
        Save a csv file summarising the cluster volumes.

        Parameters
        ----------
        folder : string
            Where to save the file.
        suffix : string, optional
            Suffix of csv filename before extension. The default is ''.

        """
        
        if self._clusters:
            
            path = f"{folder}/{self._filename}_{suffix}.csv"
            
            if self._clusters[0].get_volume() is None:
                
                print("Calculating volumes with default parameters.")
                self.calculate_volumes()
            
            with open(path, mode='w', newline='') as csv_file:
                
                writer = csv.writer(csv_file)
                writer.writerow(["cluster", "volume"]) # colnames
                
                for cluster in self._clusters:
                    writer.writerow([cluster.get_label(),
                                     cluster.get_volume()])
        
        else: print("Please run the clustering method first.")
        
    def calculate_az_dist(self, results_file=None):
        """
        Calculate azimuth and distance from plot centre, for each cluster. The
        point cloud must be transformed to have the centre at 0,0, with the y
        axis corresponding to north and the x axis corresponding to east.

        Parameters
        ----------
        results_file: string
            Where to write the results (csv file path). The default is None.

        """
        
        if results_file is None: path = 'azim_dist.csv'
        else: path = results_file
        
        # Get plot name from filename (to match the reference field)
        plot_name = self._filename
        plot_name = plot_name[:5] # get 5 first characters
        if plot_name.endswith('_'):
            plot_name = plot_name[:-1] # avoid end with _
        
        # Create results file if it does not already exist
        if not os.path.exists(path):
            
            header = ['plot', 'cluster', 'azimuth', 'distance']
            
            with open(results_file, 'w', newline='') as csv_file:
                
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header)
                
        # Calculate azimuth and distance for each cluster
        for cluster in self._clusters:
            cluster.az_dist(plot_name, path)


class Cluster:
    """
    Manage a cluster of 3D points, with methods for exporting files and
    filtering methods.

    """

    def __init__(self, label, points, volume=None):
        self._label = int(label)
        self._points = points
        self._volume = volume
        self._x = self._points[:, 0]
        self._y = self._points[:, 1]
        self._z = self._points[:, 2]
        self._filtered = False
        
    def __str__(self):
        return(f"Cluster {self._label}, {len(self._points)} points, filtering "
               + f"state: {self._filtered}")
    
    def __repr__(self):
        return (f"Cluster(label={self._label}, num_points={len(self._points)},"
                + f" filtering={self._filtered})")

    def get_label(self):
        return(self._label)

    def get_points(self):
        return(self._points)
    
    def get_volume(self):
        return(self._volume)

    def is_filtered(self, boolean=None):
        """
        Update the filtering status if boolean (True or False) is specified.
        Else, return the filtering status. A filtered cluster will be removed
        from the point cloud.

        """

        if boolean == None: return(self._filtered)

        else: self._filtered = boolean

    def set_label(self, label):
        """
        Modify the cluster label.

        """

        self._label = label
        
    def create_img(self, save_path, prefix, figsize, dpi):
        """
        Plot cluster points with a color depending on the z-coordinate.
        Optimise the view by rotating the image on the z axis to allow the
        maximum number of points to be seen. Then, save the plot as a png file
        at the specified path.

        Parameters
        ----------
        save_path : string
            Folder where to save the file.
        prefix : string
            Prefix of image name.
        figsize : tuple
            Figure size in inches.
        dpi : integer
            Dots per inch.

        """

        # Create figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')

        # Create a color gradient based on z-values
        cmap = plt.get_cmap('viridis')

        # Create a color gradient based on z-values
        cmap = plt.get_cmap('viridis')
        colors = cmap(self._z)

        # Generate graph
        ax.scatter(self._x, self._y, self._z, c=colors)

        # Hide axes
        ax.set_axis_off()

        # Calculate centroid of the point cloud
        centroid = np.mean(self.get_points(), axis=0)

        # Calculate vector from centroid to each point
        vectors = self.get_points() - centroid

        # Convert to polar coordinates
        azimuths = np.array(list(map(cartesian_to_spherical, vectors)))[:, 1]

        # Calculate average azimuth angle
        average_azimuth = np.mean(azimuths) * 180 / np.pi

        # Set the view angle based on average azimuth
        ax.view_init(azim=average_azimuth)

        # Save the figure as an image
        plt.savefig(f'{save_path}/{prefix}_cluster_{self._label}.png', dpi=dpi)

        plt.close()

    def las_points(self, header):
        """
        Create a PointRecord object with las points so that they can be added
        later to an existing las file.

        Parameters
        ----------
        header : laspy.header.LasHeader
            Same header as the file to which the points will be written.
        label : integer
            Label that will be given to points cluster field.

        Returns
        -------
        point_record : laspy.point.record.ScaleAwarePointRecord
            Points compatible with las file.

        """
        
        # Initialise point record
        point_record = laspy.ScaleAwarePointRecord.zeros(self._points.shape[0],
                                                         header=header)

        point_record.x = self._x
        point_record.y = self._y
        point_record.z = self._z
        point_record.cluster = str(self._label)
        
        if self._volume is not None:
            point_record.volume = np.array([self._volume], dtype=np.float64)

        return point_record
    
    def create_las(self, save_path, prefix, suffix):
        """
        Create a .las file containing cluster points.

        Parameters
        ----------
        save_path : string
            Folder where to save the file.
        prefix : string
            Prefix of file name.
        suffix : string
            Filename suffix before extension.
        
        """
        
        # Create destination folder if needed
        if not os.path.exists(save_path): os.makedirs(save_path)
        
        # Create new .las file
        path_out = f'{save_path}/{prefix}_{suffix}_cluster_{self._label}.las'
        new_las = laspy.create(point_format=7, file_version="1.4")
        
        # Add a new field "cluster"
        new_las.add_extra_dim(laspy.ExtraBytesParams(name='cluster',
                                                     type=np.uint32))
        if self._volume is not None:
            new_las.add_extra_dim(laspy.ExtraBytesParams(name='volume',
                                                         type=np.float64))
            
        new_las.header.scales = np.array([1.e-05, 1.e-05, 1.e-05])
        new_las.write(path_out)
        
        points = self.las_points(header=new_las.header)
        
        # Append .las points to new file
        with laspy.open(path_out, mode="a") as las_out:
            las_out.append_points(points)
    
    def nb_points_filter(self, nb_points):
        """
        Filter a cluster based on a minimum number of points.

        Parameters
        ----------
        nb_points : integer
            Minimum number of points a cluster must contain.

        """

        if nb_points is not None and not self.is_filtered():

            if len(self.get_points()) < nb_points:
                self.is_filtered(True)

    def distance_from_centre_filter(self, plot_name, distance,
                                    coord_file, sep, dec):
        """
        Check whether most of the cluster is located within a circle of radius
        "distance" from the centre of the plot. The centre coordinates must be
        in coord_file (infos needed: reference, Xcentre, Ycentre).

        Parameters
        ----------
        plot_name : string
            Plot reference, has to be the same as in coord_file.
        distance : integer
            Actual radius of the inventory plot.
        coord_file : string
            Path leading to the csv file that contains coordinates of the plot
            centre. If None, the filter is ignored. The default is None.
        sep : string
            Seperator of the csv file.
        dec : string
            Decimal separator of the csv file.

        """
        
        if distance is not None and not self.is_filtered():

            # Get xy coordinates only
            points_xy = self._points[:, :2]
            
            centre_xy = np.array([0,0])
            
            if coord_file is not None:
                
                df = pd.read_csv(coord_file, sep=sep, decimal=dec)
    
                # Get centre coordinates
                centre_xy = np.array(df.loc[df['reference'] == plot_name,
                                            ['Xcentre', 'Ycentre']])

            # Calculate euclidean distances between each point and the centre
            distances = cdist(points_xy, centre_xy, 'euclidean')[0]
            
            if len(distances) > 0: # avoiding errors with np.mean
                
                # Calculate the proportion of points within 18m of the centre
                prop = np.mean(distances < distance)
            
                if prop < 0.5: self.is_filtered(True)
    
    def flying_filter(self, delta):
        """
        Filter "flying" branches whose lowest point is more than delta metres
        from the ground.

        Parameters
        ----------
        delta : float
            Maximum distance from the ground.

        """
        
        if delta is not None and not self.is_filtered():

            # Lowest z-value
            min_z = np.min(self._points[:, 2])

            if min_z > delta:
                self.is_filtered(True)

    def length_filter(self, min_dist):
        """
        Filter based on the minimum total length of the cluster.

        Parameters
        ----------
        min_dist: integer
            Minimum distance that the 2 furthest points of the cluster must be
            from each other. Set to None to ignore length filtering.

        """

        if min_dist is not None and not self.is_filtered() and \
            len(self.get_points()) < 50000:  # avoid unnecessary calculations
            
            try:
                
                # Calculate distances between all points
                distances = cdist(self._points, self._points, 'euclidean')

                # Maximum length of the shape
                max_dist = np.max(distances)
                
                # Filter out clusters that are too small
                if max_dist < min_dist:
                    self.is_filtered(True)
            
            except MemoryError: print("MemoryError: Skipping cluster due to " +
                                      "excessive memory usage.")
    
    def calculate_volume(self, alpha, plot):
        """
        Create an alpha shape (concave hull) from the cluster points and
        calculate its volume.
        
        Parameters
        ----------
        alpha: integer
            Alpha parameter for alpha shape creation. The higher the alpha
            value is, the tighter the bounding shape will fit the data.
        plot: boolean
            If True, a plot of the alpha shape is shown.
        
        """
        
        alpha_shape = alphashape.alphashape(self._points, int(alpha))
        self._volume = alpha_shape.volume
        
        # Show a plot of the alpha shape
        if plot is True:
            
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces)
            plt.show()
    
    def az_dist(self, plot_name, results_file):
        """
        Calculate azimuth and distance of the middle of the cluster, from the
        centre of the plot (0,0). Write the results in a csv file.

        Parameters
        ----------
        plot_name : string
            The plot reference.
        results_file : string
            Where to write the results.

        """
        
        # Get x and y coordinates of cluster centroïd
        x_cluster = sum(self._points[:, 0])/len(self._points[:, 0])
        y_cluster = sum(self._points[:, 1])/len(self._points[:, 1])
        
        # Calculate azimuth
        azimuth = np.degrees(np.arctan2(x_cluster, y_cluster)) % 360
        
        # Calculate distance
        distance = np.sqrt(x_cluster**2 + y_cluster**2)
        
        # Write results
        with open(results_file, 'a', newline='') as csv_file:
            
            csv_writer = csv.writer(csv_file)
            new_line = [plot_name, self.get_label(), azimuth, distance]
            csv_writer.writerow(new_line)
        

class ClusteringTuner(BaseEstimator):
    """
    Hyperparameter tuner for DBSCAN and HDBSCAN clustering algorithms.
    
    """
    
    def __init__(self, cluster_type, n_iter=10):
        """
        Initialize the ClusteringTuner object.

        Parameters
        ----------
        cluster_type : string, optional
            The type of clustering algorithm to tune. Possible values are
            'dbscan' and 'hdbscan'.
        n_iter : integer, optional
            The number of iterations for randomized search. The default is
            10.
                
        """
        
        self.cluster_type = cluster_type
        self.n_iter = n_iter
        self.best_params = None

    def fit(self, X):
        """
        Fit the tuner to the input data and find the best hyperparameters for
        the specified cluster_type.

        Parameters
        ----------
            X : array-like, shape (n_samples, n_features)
                The input data.
            
        """
        
        if self.cluster_type == 'dbscan':
            
            param_dist = {
                'eps': [0.03, 0.04, 0.05, 0.06],
                'min_samples': [10, 20, 50, 100, 200]
            }
            
            # Filter out UserWarning during tuning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.best_params_ = self._tune_params(DBSCAN(), param_dist, X)
            
            self.best_clustering_ = DBSCAN(**self.best_params_)

        elif self.cluster_type == 'hdbscan':
            
            param_dist = {
                'min_cluster_size': [100, 200, 500, 1000, 1500],
                'max_cluster_size': [150000, 200000, 250000, 300000],
                'min_samples': [2, 5, 10, 20, 50]
            }
                        
            # Filter out UserWarning during tuning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.best_params_ = self._tune_params(HDBSCAN(), param_dist, X)
            
            self.best_clustering_ = HDBSCAN(**self.best_params_)

        else:
            raise ValueError(
                "Invalid cluster_type. Use 'dbscan' or 'hdbscan'.")

        self.best_clustering_.fit(X)
    
    def _silhouette_score(self, estimator, X):
        """
        Calculate the silhouette score for the clustering estimator.

        Parameters
        ----------
        estimator : object
            The clustering model to evaluate.
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        silhouette_score : float
            The silhouette score of the clustering.

        """
        labels = estimator.fit_predict(X)
        return silhouette_score(X, labels)
    
    def _tune_params(self, clustering_model, param_dist, X):
        """
        Perform randomized search for hyperparameter tuning.

        Parameters
        ----------
        clustering_model : object
            The clustering model to tune.
        param_dist : dict
            Dictionary of hyperparameter distributions for randomized search.
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        best_params : dict
            Dictionary of the best hyperparameters found during tuning.

        """
        
        #/!\/!\/!\ scorers need to be changed, it doesn't work yet /!\/!\/!\
        if self.cluster_type == 'dbscan':
            scorer = make_scorer(self._silhouette_score,
                                 greater_is_better=True)
        else:
            scorer = make_scorer(validity.validity_index,
                                 greater_is_better=True)
        
        tuner = RandomizedSearchCV(clustering_model,
                                   param_distributions=param_dist,
                                   n_iter=self.n_iter, n_jobs=4,
                                   scoring=scorer,
                                   random_state=42)
        tuner.fit(X)
        
        self.best_params = tuner.best_params_
        
        return self.best_params

    def fit_predict(self, X):
        """
        Fit the tuner to the input data and perform clustering with the best
        hyperparameters.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        
        Returns
        ----------
        labels : array, shape (n_samples,)
            Cluster labels for each data point.
        
        """
        
        self.fit(X)
        return self.best_clustering_.labels_
    
    def get_best_params(self): return self.best_params


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


def label_from(image):
    """
    Return the cluster label from the cluster image.

    Parameters
    ----------
    image : string
        Image path.

    """
    
    # Filename
    basename = os.path.basename(image)
    
    start_index = basename.find("_cluster_") + len("_cluster_")
    end_index = basename.find(".png")
    
    return basename[start_index:end_index]