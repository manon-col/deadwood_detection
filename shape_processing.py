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
from scipy.spatial.distance import cdist
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler


class shape_processing:
    """
    Manage a point cloud representing an (imperfect) cylinder shape contained
    in a .txt file (cloud ASCII), with different filtering methods. The file
    must contain all coordinates and have headers as column titles.
    
    """
    
    def __init__(self, file):
        
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
        
        print("File "+self._filename+".txt loaded successfully.")
    
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
            
            print("Angle: "+str(abs_angle))       
    
    def distance_from_centre(self, plot_name, distance=18):
        """
        Check whether most of the object is located within a circle of radius
        "distance" from the centre of the plot. The centre coordinates are in
        the file "spheres_coordinates.csv".
        
        Parameters
        ----------
        plot_name : string
            Plot reference, has to be the same as in the sphere_coordinates
            file.
        distance : integer, optional
            Actual radius of the inventory plot. The default is 18m.
        
        """
        
        if self._filtered == False:
            
            # Get xy coordinates only
            points_xy = self._xyz[:, :2]
            
            df = pd.read_csv('spheres_coordinates.csv', sep=';')
            centre_xy = np.array(df.loc[df['reference'] == plot_name,
                                     ['Xcentre', 'Ycentre']])
                        
            # Calculate euclidean distances between each point and the centre
            distances = cdist(points_xy, centre_xy, 'euclidean')[0]
            
            # Calculate the percentage of points within 18m of the centre
            percentage = np.mean(distances < 18) * 100
            
            if percentage < 50: self._filtered = True
            
            print("Percentage of points in the plot: "+str(percentage))
    
    def flying_filter(self, delta = 0.05):
        """
        Filter "flying" branches whose lowest point is more than delta metres
        from the ground. Works better if the points have an altitude of 0...
        
        Parameters
        ----------
        delta : float, optional
            Maximum distance from the ground. The default is 0.05m.
            
        """
        
        # Lowest z-value
        min_z = np.min(self._xyz[:, 2])
        
        if min_z > delta: self._filtered = True
        
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
            
            print("Max length of the shape: "+str(max_distance))
    
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
            
            print('Angular deviation: '+str(angular_diff))
    
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
            
            print("Variance: "+str(variance))
    
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
        new_las.header.scales = np.array([1.e-05, 1.e-05, 1.e-05])
        new_las.write(path_out)
        
        # Browsing all .txt files
        for file in glob.glob(folder+'/*.txt'):
            
            ## Shape processing

            sh = shape_processing(file)
            
            # Filtering according to inclination to vertical axis
            sh.inclination_filter()
            
            # Filtering "flying" branches
            sh.flying_filter()
            
            # Filtering shapes outside the inventory plot
            sh.distance_from_centre(plot_name=folder_name)
            
            # Filter based on the resemblance of the projection to a deformed
            # rectangle
            # sh.rectangle_filter()
            
            # Filtering according to the distribution of points along main axis
            sh.stable_density_filter(treshold=0.01)
            
            # Filtering according to shape length
            sh.length_filter()
            
            if sh._filtered == True :
                print("Filtered. \n")
                
            else:
                print("Unfiltered. \n")
                remain += 1
                
                # Create las points
                points = sh.las_points(header=new_las.header, label=remain)
                
                # Append las points to new file
                with laspy.open(path_out, mode="a") as las_out:
                    las_out.append_points(points)
            
            total += 1
            
        print("Done!\n"+str(remain)+" shapes remaining out of "+str(total)\
              +".")