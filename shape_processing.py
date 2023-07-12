# -*- coding: utf-8 -*-
"""
Program designed to filter cylindrical shapes (point cloud in .txt format such
as ascii cloud exported from CloudCompare) according to various statistical
criteria.

@author: manon-col
"""


import os
import laspy
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler


class shape_processing:
    """
    Manage a point cloud representing an (imperfect) cylinder shape contained
    in a .txt file (cloud ASCII), with different filtering methods. The file
    must contain all coordinates and have headers as column titles.
    
    """
    
    def __init__(self, file, verbose=False):
        
        # File containing shape data
        self._file = file
        # Filename without extension
        self._filename = os.path.splitext(os.path.basename(self._file))[0]
        # Reading file
        self._data = pd.read_csv(self._file, delimiter=' ', header=0)
        # Extracting coordinates
        self._xyz = self._data[['//X', 'Y', 'Z']].values
        # Filtering state
        self._filtered = False
        
        self._verbose = verbose
        
        if self._verbose:
            print("File "+self._filename+".txt loaded successfully.")
    
    def is_filtered(self):
        """
        Return the shape filtering state (boolean).

        """
        
        if self._verbose: print("Filtering status of "+self._filename+".txt: "\
                                +self._filtered)
        return self._filtered
    
    def inclination_filter(self, angle=45):
        """
        Calculate shape inclination to the vertical axis and applicate a filter
        based on angle value.

        Parameters
        ----------
        angle : integer, optional
            Maximum inclination the shape must have (regarding vertical ax).
            The default is 45. Note: angle of 0° corresponds to a vertical
            shape, angle of 90° (max value) correponds to an horizontal shape.

        """
        
        if self._filtered == False:
            
            # Perform PCA to find the principal components
            pca = PCA(n_components=3)
            pca.fit(self._xyz)
        
            # Get the first principal component (estimated axis of the shape)
            estimated_axis = pca.components_[0]
        
            # Compute the inclination angle of the estimated axis
            dot_product = np.dot(estimated_axis, np.array([0, 0, 1]))
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)
            
            # Adjusting the angle to be in the range [0, 90]
            adjusted_angle = angle_deg if angle_deg <= 90 else 180 - angle_deg
        
            # Check if the absolute angle is greater than the specified angle
            abs_angle = abs(adjusted_angle)
            if abs_angle < angle: self._filtered = True
            
            if self._verbose: print("Angle: "+str(abs_angle))       
    
    def distance_from_centre(self, plot_name, coord_file, sep=';', dec=',',
                             distance=18):
        """
        Check whether most of the object is located within a circle of radius
        "distance" from the centre of the plot. The centre coordinates must be
        in coord_file (infos needed: reference, Xcentre, Ycentre).
        
        Parameters
        ----------
        plot_name : string
            Plot reference, has to be the same as in coord_file.
        coord_file : string
            Path leading to the csv file that contains coordinates of the plot
            centre.
        sep : string, optional
            Seperator of the csv file. The default is ';'.
        dec : string, optional
            Decimal separator of the csv file. The default is ','.
        distance : integer, optional
            Actual radius of the inventory plot. The default is 18m.
        
        """
        
        if self._filtered == False:
            
            # Get xy coordinates only
            points_xy = self._xyz[:, :2]
            
            df = pd.read_csv(coord_file, sep=sep, decimal=dec)
            
            # Get centre coordinates
            centre_xy = np.array(df.loc[df['reference'] == plot_name,
                                     ['Xcentre', 'Ycentre']])
            
            # Calculate euclidean distances between each point and the centre
            distances = cdist(points_xy, centre_xy, 'euclidean')[0]
            
            # Calculate the percentage of points within 18m of the centre
            percentage = np.mean(distances < distance) * 100
            
            if percentage < 50: self._filtered = True
            
            if self._verbose:
                print("Percentage of points in the plot: "+str(percentage))
    
    def flying_filter(self, delta = 0.05):
        """
        Filter "flying" branches whose lowest point is more than delta metres
        from the ground.
        
        Parameters
        ----------
        delta : float, optional
            Maximum distance from the ground. The default is 0.05m.
            
        """
        
        # Lowest z-value
        min_z = np.min(self._xyz[:, 2])
        
        if min_z > delta:
            
            self._filtered = True
        
            if self._verbose: print("Too high above the ground.")
        
    def length_filter(self, length=1):
        """
        Filter based on the minimum total length of the shape.

        Parameters
        ----------
        length : float, optional
            Minimum length. The default is 1m.

        """
        
        if self._filtered == False:
            
            # Calculate distances between all points
            distances = cdist(self._xyz, self._xyz)
            
            # Maximum length of the shape
            max_distance = np.max(distances)
            
            if max_distance < length: self._filtered = True
            
            if self._verbose: print("Max length of the shape: "+ \
                                    str(max_distance))
    
    def rectangle_filter(self, treshold=3, visualisation=False):
        """
        Filter based on the detection of deformed rectangles.

        Parameters
        ----------
        treshold : float, optional
            Maximum angular deviation tolerated. The default is 3.
        visualisation : bool, optional
            If True, a plot of projected PCA points is created. The default is
            False.

        """
        
        if self._filtered == False:
            
            # Centre of mass of the points cloud
            centre = np.mean(self._xyz, axis=0)
    
            # Center the data around zero
            centered_points = self._xyz - centre
            
            # Principal component analysis
            pca = PCA()
            pca.fit(centered_points)
            
            # Project points onto the first two principal components
            projected_points = pca.transform(centered_points)[:, :2]
            
            if visualisation == True:
                
                plt.scatter(projected_points[:, 0], projected_points[:, 1])
                plt.xlabel('1st principal component')
                plt.ylabel('2nd principal component')
                plt.title('Points projection after PCA, file '+self._filename)
                plt.show()
            
            # Calculate angular difference between the eigenvectors and the
            # assumed principal axis
            first_eigenvector = pca.components_[0]
            angle = np.arctan2(first_eigenvector[1], first_eigenvector[0])
            angular_diff = np.abs(angle - np.pi/2)  # Compare with angle pi/2
    
            # Checking if a deformed rectangle is detected
            if angular_diff > treshold: self._filtered = True
            
            if self._verbose: print('Angular deviation: '+str(angular_diff))
    
    def stable_density_filter(self, treshold=0.025):
        """
        Filter based on the fact that, for a cylindrical shape, the density of
        points is less variable along the axis than for irregular shape.

        Parameters
        ----------
        treshold : float, optional
            Maximum variance of the points projected onto the 2nd principal PCA
            axis that is tolerated. The default is 0.025.

        """
                
        if self._filtered == False:
            
            # Centre of mass of the points cloud
            centre = np.mean(self._xyz, axis=0)
    
            # Center the data around zero
            centered_points = self._xyz - centre
            
            # Principal component analysis
            pca = PCA()
            pca.fit(centered_points)
            
            # Project points onto the 2nd principal ax
            projected_points = pca.transform(centered_points)[:, 2]
            
            # Calculate projected points variance
            variance = np.var(projected_points)
            
            if variance > treshold: self._filtered = True
            
            if self._verbose: print("Variance: "+str(variance))
    
    def ellipticity_filter(self, treshold=0.4):
        """
        Calculate shape ellipticity and applicate a filter based on a treshold.
        WARNING: the results showed that this method unreliable.

        Parameters
        ----------
        treshold : float, optional
            Minimum ellipticity value the shape must have. The default is 0.4.

        """
        
        if self._filtered == False:
            
            # Perform PCA to find the new projection axis
            pca = PCA(n_components=3)
            pca.fit(self._xyz)
            
            # Project the points onto the new axis
            projected_points = pca.transform(self._xyz)[:, :2]
            
            # Compute fitted ellipse on the projected points
            scaler = StandardScaler()
            scaled_points = scaler.fit_transform(projected_points)
            
            covariance_estimator = MinCovDet()
            covariance_estimator.fit(scaled_points)
            covariance_matrix = \
                scaler.inverse_transform(covariance_estimator.covariance_)
            _, eigenvalues, _ = np.linalg.svd(covariance_matrix)
            major_axis = np.sqrt(eigenvalues[0])
            minor_axis = np.sqrt(eigenvalues[1])
            ellipticity = minor_axis / major_axis
            
            # Filtering
            if ellipticity < treshold: self._filtered = True
                    
            if self._verbose: print("Ellipticity: "+str(ellipticity))
    
    def save(self, folder):
        """
        If the cylinder is not filtered, save a copy of the cylinder file in a
        specified folder.

        Parameters
        ----------
        folder : path
            Path of the folder where the saved files are.
            
        """
        
        if self._filtered == False:
            
            dest = folder+'/'+self._filename+'.txt'
            shutil.copy2(self._file, dest)
    
    def las_points(self, header, label):
        """
        Convert ASCII file points into las points (PointRecord object) so that
        they can be added later to an existing las file. Keeping only xyz and
        classification fields.

        Parameters
        ----------
        header : laspy.header.LasHeader
            Same header as the file to which the points will be written.
        label : integer
            Label that will be given to points classification field. It should
            correspond to the filtered shape number.

        Returns
        -------
        point_record : laspy.point.record.ScaleAwarePointRecord
            Points compatible with las file.

        """
        
        # Initialise point record
        point_record = laspy.ScaleAwarePointRecord.zeros(self._data.shape[0],
                                                         header=header)
        
        point_record.x = self._data['//X']
        point_record.y = self._data['Y']
        point_record.z = self._data['Z']
        point_record.classification = str(label)
                    
        return point_record