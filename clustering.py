# -*- coding: utf-8 -*-
"""
Program for clustering points from a .las file, filtering, and managing
clustering outputs.

@author: manon-col
"""


import os
import laspy
import random
import numpy as np
import pandas as pd
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
        # List of clusters
        self._clusters = []
        # List of unfiltered clusters = clusters to save
        self._unfiltered = []
        # Filename without extension
        self._filename = os.path.splitext(os.path.basename(self._las_file))[0]
        
        print(f"File {self._filename}.las loaded successfully.")

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
          
        print(f"Clustering {self._filename}.las points...")
        
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
        Draw the clustering results (simply), with points in the same cluster
        of the same colour.
        
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
        
    def filtering(self,
                  nb_points=500,
                  coord_file=None,
                  sep=';',
                  dec=',',
                  distance_from_centre=18,
                  delta=0.05,
                  min_dist=1):
        """
        Basic cluster filter based on a minimum number of points and a minimum
        maximal length.

        Parameters
        ----------
        
        nb_points : integer, optional
            Minimum number of points a cluster must contain. Set to None to
            ignore point density filtering. The default is 500.
        coord_file : string, optional
            Path leading to the csv file that contains coordinates of the plot
            centre. Set to None to ignore distance_from_centre filtering. The
            default is None.
        sep : string, optional
            Seperator of the csv file. The default is ';'.
        dec : string, optional
            Decimal separator of the csv file. The default is ','.
        distance_from_centre : integer, optional
            Actual radius of the inventory plot. The default is 18m.
        delta : float, optional
            Maximum distance from the ground. The default is 0.05m.    
        min_dist: integer, optional
            Minimum distance that the 2 furthest points of the cluster must be
            from each other. Set to None to ignore length filtering. The
            default is 1m.
            
        """
        
        print("Filtering clusters...")
        
        if self._clusters:
            
            for cluster in self._clusters:
                                    
                cluster.noise_filter()
                cluster.nb_points_filter(nb_points=nb_points)
                cluster.distance_from_centre_filter(plot_name=self._filename,
                                                    distance=distance_from_centre,
                                                    coord_file=coord_file,
                                                    sep=sep,
                                                    dec=dec)
                cluster.flying_filter(delta=delta)
                cluster.length_filter(min_dist=min_dist)
                
            self._make_unfiltered_list()
            
            print(f"{len(self._unfiltered)} clusters remaining out of " +\
                  f"{len(self._clusters)}.")
        
        else: print("Please run the clustering method first.")
    
    def reset_filtering(self):
        """
        Set the status of all clusters to unfiltered.
        
        """
        
        for cluster in self._clusters: cluster.is_filtered(False)
    
    def _make_unfiltered_list(self):
        """
        Make the list of clusters to save.

        """
        
        for cluster in self._clusters:
            if not cluster.is_filtered(): self._unfiltered.append(cluster)
        
        # Relabel clusters
        for index in range(len(self._unfiltered)):
            self._unfiltered[index].set_label(index+1)
    
    def save_clusters_las(self, folder):
        """
        Save the clustering results in a new las file, in the specified folder,
        with the cluster label in the 'cluster' field (noise is in "0"
        category).

        """
        
        print("Saving unfiltered clusters in .las file...")
        
        if self._clusters:
            
            # Create new .las file
            path_out = f'{folder}/{self._filename}_clusters.las'
            new_las = laspy.create(point_format=7, file_version="1.4")
            # Adding a new field "cluster"
            new_las.add_extra_dim(laspy.ExtraBytesParams(name="cluster",
                                                         type=np.uint16))
            new_las.header.scales = np.array([1.e-05, 1.e-05, 1.e-05])            
            new_las.write(path_out)
            
            # Filling the "cluster" field
            
            for cluster in self._unfiltered:
                
                points = cluster.las_points(header=new_las.header)
                
                # Append .las points to new file
                with laspy.open(path_out, mode="a") as las_out:
                    las_out.append_points(points) 
            
            if not self._unfiltered:
                
                print("Warning: clusters are not filtered, saving all "+\
                      "clusters...")
                
                for cluster in self._clusters:
                                        
                    points = cluster.las_points(header=new_las.header)
                    
                    # Append .las points to new file
                    with laspy.open(path_out, mode="a") as las_out:
                        las_out.append_points(points) 
            
            print(f"Clustering results successfully saved in {path_out}.")
        
        else: print("Please run the clustering method first.")
    
    def save_clusters_img(self, folder, figsize=(4,4), dpi=75):
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
        
        if self._unfiltered:
            
            print("Saving unfiltered clusters in .png files...")
            
            for cluster in self._unfiltered:
                    
                save_path = f'{folder}/{self._filename}'
                
                # Create destination folder if needed
                if not os.path.exists(save_path): os.makedirs(save_path)
                
                cluster.create_img(save_path=save_path,
                                   prefix=self._filename,
                                   figsize=figsize, dpi=dpi)
            
            print(f"Images successfully saved in {save_path}.")
        
        else: print("Please filter clusters first.")


class Cluster:
    """
    Manage a cluster of 3D points, with methods for exporting files and
    filtering methods.
    
    """
    
    def __init__(self, label, points):
        self._label = int(label)
        self._points = points
        self._x = self._points[:,0]
        self._y = self._points[:,1]
        self._z = self._points[:,2]
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
        normalize = plt.Normalize(vmin=min(self._z), vmax=max(self._z))
        colors = cmap(normalize(self._z))
        
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
                    
        return point_record
    
    def noise_filter(self):
        """
        Filter a cluster if it is noise.
        
        """
        
        if self.get_label() == 0: self.is_filtered(True)
        
    def nb_points_filter(self, nb_points):
        """
        Filter a cluster based on a minimum number of points.

        Parameters
        ----------
        nb_points : integer
            Minimum number of points a cluster must contain.

        """
        
        if nb_points is not None and not self.is_filtered():
        
            if len(self.get_points()) < nb_points: self.is_filtered(True)
    
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
        
        if coord_file is not None and not self.is_filtered(): 
            
            # Get xy coordinates only
            points_xy = self._points[:, :2]
            
            df = pd.read_csv(coord_file, sep=sep, decimal=dec)
            
            # Get centre coordinates
            centre_xy = np.array(df.loc[df['reference'] == plot_name,
                                        ['Xcentre', 'Ycentre']])
            
            # Calculate euclidean distances between each point and the centre
            distances = cdist(points_xy, centre_xy, 'euclidean')[0]
            
            # Calculate the percentage of points within 18m of the centre
            percentage = np.mean(distances < distance) * 100
            
            if percentage < 50: self.is_filtered(True)
    
    def flying_filter(self, delta=None):
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
            
            if min_z > delta: self.is_filtered(True)
    
    def length_filter(self, min_dist):
        """
        Filter based on the minimum total length of the cluster.

        Parameters
        ----------
        min_dist: integer, optional
            Minimum distance that the 2 furthest points of the cluster must be
            from each other. Set to None to ignore length filtering. The
            default is 1m.

        """
        
        if min_dist is not None and not self.is_filtered():
            
            try:
            
                # Calculate distances between all points
                distances = cdist(self._points, self._points, 'euclidean')
                
                # Maximum length of the shape
                max_dist = np.max(distances)
                
                if max_dist < min_dist: self.is_filtered(True)
            
            except MemoryError:
                
                print("MemoryError: Skipping cluster due to "+\
                      "excessive memory usage.")


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

pla = ClEngine('computree_outputs/BM08_ct.las')
pla.DBSCAN_clustering()
pla.filtering(nb_points=500,
              coord_file='preprocessing/spheres_coordinates.csv',
              sep=';',
              dec=',',
              distance_from_centre=18,
              delta=0.05,
              min_dist=None)
# pla.save_clusters_img('clusters_img')
pla.save_clusters_las('clusters_las')