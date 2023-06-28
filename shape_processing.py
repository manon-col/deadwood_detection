# -*- coding: utf-8 -*-
"""
Program filtering detected cylinders (for example: cloud ASCII files exported
from CloudCompare) according to their orientation. Cylinders detected in each
plot are in the "cylinders_raw" folder and outputs are in the
"cylinders_filtered" folder.

@author: manon-col
"""


import os
import glob
import laspy
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
# from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler


class shape_processing:
    """
    Manage a point cloud representing an (imperfect) cylinder model contained
    in a .txt file (cloud ASCII). The file must contain all coordinates and
    already computed normals.
    
    """
    
    def __init__(self, file):
        
        # File containing cylinder data
        self._file = file
        # Filename without extension
        self._filename = os.path.splitext(os.path.basename(self._file))[0]
        # Reading file
        self._data = pd.read_csv(self._file, delimiter=' ', header=0)
        # Extracting coordinates
        self._xyz = self._data[['//X', 'Y', 'Z']].values
        # Extraction normals
        self._normals = self._data[['Nx', 'Ny', 'Nz']].values
        # Filtering state
        self._filtered = False
        
        print("File "+self._filename+".txt loaded successfully.")
        
    def orientation_filter(self, angle=45):
        """
        Calculate shape orientation and applicate a filter.

        Parameters
        ----------
        angle : integer, optional
            Minimum inclination the shape must have. The default is 45.

        """
                
        # Calculating the mean vector of the normals
        mean_vector = np.mean(self._normals, axis=0)
        
        # Normalizing the orientation vector
        normalized_vector = mean_vector/np.linalg.norm(mean_vector)
    
        # Computing the dot product between the normalized vector and the
        # vertical axis (0, 0, 1)
        dot_product = np.dot(normalized_vector, np.array([0, 0, 1]))
    
        # Computing the angle between the vectors in radians
        angle_rad = np.arccos(dot_product)
    
        # Converting the angle to degrees, handling cases where the angle is
        # greater than 90 degrees
        angle_deg = np.degrees(angle_rad) if angle_rad <= np.pi/2 else 180 - \
            np.degrees(angle_rad)
        
        # Checking if the absolute angle is greater than the specified angle
        abs_angle = abs(angle_deg)
        
        if abs_angle > angle: self._filtered = True
        
        print("Angle: "+str(abs_angle))
    
    def rectangle_filter(self, treshold=3, visualisation=False):
        """
        Filter based on the detection of deformed rectangles.

        Parameters
        ----------
        treshold : integer, optional
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
                plt.title('Points projection after PCA '+self._filename)
                plt.show()
            
            # Calculate angular difference between the eigenvectors and the
            # assumed principal axis
            first_eigenvector = pca.components_[0]
            angle = np.arctan2(first_eigenvector[1], first_eigenvector[0])
            angular_diff = np.abs(angle - np.pi/2)  # Compare with angle pi/2
    
            # Checking if a deformed rectangle is detected
            if angular_diff > treshold: self._filtered = True
            
            print('Angular deviation: '+str(angular_diff))
    
    def stable_density_filter(self, treshold=0.04):
        """
        Filter based on the fact that, for a cylindrical shape, the density of
        points is less variable along the axis than for irregular shape.

        Parameters
        ----------
        treshold : integer, optional
            Maximum variance of the points projected onto the 2nd principal PCA
            axis that is tolerated. The default is 0.04.

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
            
            print("Variance: "+str(variance))
    
    def ellipticity_filter(self, treshold=0.97):
        """
        Calculate shape ellipticity and applicate a filter based on a treshold.

        Parameters
        ----------
        treshold : integer, optional
            Minimum ellipticity value the shape must have. The default is 0.97.

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
            covariance_matrix = scaler.inverse_transform(covariance_estimator.covariance_)
            _, eigenvalues, _ = np.linalg.svd(covariance_matrix)
            major_axis = np.sqrt(eigenvalues[0])
            minor_axis = np.sqrt(eigenvalues[1])
            ellipticity = minor_axis / major_axis
            
            # Filtering
            if ellipticity < treshold: self._filtered = True
                    
            print("Ellipticity: "+str(ellipticity))
    
    def save(self, folder):
        """
        If the cylinder is not filtered, save a copy of the cylinder file in a
        specified folder.

        Parameters
        ----------
        folder : path
            Path of the folder where the concerned files are.
            
        """
        
        if self._filtered == False:
            dest = folder+'/'+self._filename+'.txt'
            shutil.copy2(self._file, dest)
    
    def las_points(self, header):
        """
        Convert ASCII file points into las points (PointRecord object) so that
        they can be added later to an existing las file. Keeping only xyz and
        classification fields.

        Parameters
        ----------
        header : laspy.header.LasHeader
            Same header as the file to which the points will be written.

        Returns
        -------
        point_record : laspy.point.record.ScaleAwarePointRecord
            Points compatible with las file.

        """
        
        # Initialise point record
        point_record = laspy.ScaleAwarePointRecord.zeros(self._data.shape[0],
                                                         header=header)
        
        #### Not working already!
        coeff = 1000 # to avoid a strange scaling problem occuring during saving
        
        for field in self._data:
            
            # Not clean, but avoid problem of dimension compatibility
            if field == '//X':
                point_record['X'] = self._data[field]*coeff
            
            if field == 'Y':
                point_record['Y'] = self._data[field]*coeff
                
            if field == 'Z':
                point_record['Z'] = self._data[field]*coeff
            
            if field == "Classification":
                point_record['classification'] = self._data[field]
                
            else:
                try:
                    point_record[field] = self._data[field]
                
                except Exception:
                    pass # do nothing
                    
        return point_record


def compute_volume(file):
        
    shape = PyntCloud.from_file(file, delimiter=' ', header=0, names=["x","y","z"])
    shape.plot()
    convex_hull_id = shape.add_structure("convex_hull")
    convex_hull = shape.structures[convex_hull_id]
    volume = convex_hull.volume
    print(volume)
    return(volume)


# =============================================================================
# Main program
# =============================================================================


path_raw = 'shapes_raw' # path with all raw shapes in different folders/plot
path_filtered = 'shapes_filtered' # results location
filtered = glob.glob(path_filtered+'/*.las')
filtered_names = [os.path.splitext(os.path.basename(file))[0] for file in\
                  filtered]

# Browse all folders with shapes to process
for folder in glob.glob(path_raw+'/*'):
    
    folder_name = os.path.splitext(os.path.basename(folder))[0]
    
    # Check if processing is not already done
    if os.path.splitext(os.path.basename(folder))[0]+'_cyl_filtered' not in \
        filtered_names:
        
        # Initialise counters...
        total = 0 # of total number of processed shapes
        remain = 0 # of unfiltered = remaining shapes
        
        # Initialise new las file
        path_out = path_filtered+'/'+folder_name+ '_cyl_filtered.las'
        new_las = laspy.create(point_format=7, file_version="1.4")
        new_las.write(path_out)
        
        # Browsing all .txt files
        for file in glob.glob(folder+'/*.txt'):
            
            # Shape processing
            sh = shape_processing(file)
            sh.orientation_filter(angle=60)
            # sh.ellipticity_filter(treshold=0.4)
            # sh.rectangle_filter(treshold=4, visualisation=True)
            sh.stable_density_filter()
            
            if sh._filtered == True :
                print("Filtered. \n")
                
            else:
                print("Unfiltered. \n")
                remain += 1
                
                # Create las points
                points = sh.las_points(header=new_las.header)
                
                # Append las points to new file
                with laspy.open(path_out, mode="a") as las_out:
                    las_out.append_points(points)
            
            total += 1
            
        print("Done!\n"+str(remain)+" shapes remaining out of "+str(total)\
              +".")