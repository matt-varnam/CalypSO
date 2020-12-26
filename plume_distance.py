# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:58:32 2020

@author: mbexwmv3
"""
from math import radians, cos, sin, asin, atan2, acos, sqrt, pi

#Import numpy for the great maths functions
import numpy as np

#import matplotlib for the great plotting functions
import matplotlib.pyplot as plt

#from mpl_toolkits.basemap import Basemap

#haversine calculates the distance between two latitude and longitude points
def haversine(lon1, lat1, lon2, lat2):
    '''
    Function to calculate the displacement and bearing between two GPS corrdinates
    
    INPUTS
    ------
    lon1, lat1: longitude and latitude of first point
    lon2, lat2: longitude and latitude of second point
    
    OUTPUTS
    -------
    dist: distance between two points in meters
    bearing: bearing between points (0 - 2pi clockwise from North)
    '''

    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1    
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a))
    
    # Earth radius in meters
    r = 6371000
    
    # Calculate distance
    dist = c * r
    
    # Calculate the bearing
    bearing = np.arctan2(np.sin(dlon) * np.cos(lat2), 
                    np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlat))
                    
    # Convert barings to the range (0, 2pi) instead of (-pi, pi)
    if bearing < 0:
        bearing = 2 * np.pi + bearing
    
    return dist, bearing

#Define camera parameters
fov_d = 26.5
res_x = 1600
res_y = 1200
central_pix = res_x/2

#Add GPS locations for plume, volcano and observation
plume_lats = (11.9696109329999,11.9659407169999,11.966038025295,11.977921017000,11.9730572419699,11.975561168)
plume_lons = (-86.218555516615,-86.21635728309,-86.216849400295,-86.2196001746949,-86.21832545,-86.219205945)
loc_plume  = (np.mean(plume_lons),np.mean(plume_lats))
loc_volc   = (-86.169058,11.984104)
loc_obser  = (-86.17946,12.02446)

#Define location of volcano
#volc_xpix = 253
dir_imcentre_d = 173.37931117429588 
dir_imcentre = np.radians(dir_imcentre_d)

#Calculate number of radians for fov
fov = (fov_d/180)*np.pi

#Calculate number of pixels of various points from the centre
#volc_xdev = volc_xpix - central_pix

#Calculate distance and angles to the volcano and plume
d_volc,dir_volc = haversine(loc_obser[0],loc_obser[1],loc_volc[0],loc_volc[1])
d_plume,dir_plume = haversine(loc_volc[0],loc_volc[1],loc_plume[0],loc_plume[1])

## Method one - Use known pixel containing volcanic vent to calibrate
#Calculate angle to volcano
#a_volc = np.arctan(np.divide(volc_xdev,central_pix)*np.tan(fov/2))

#Calculate angle between plume and image plane
#dir_imcentre = dir_volc - a_volc

# Method two - Use known angle to centre of image
a_volc = dir_volc - dir_imcentre

#Calculate distance to plane that intersects volcano
d_volcplane = d_volc * np.cos(abs(a_volc))

#Calculate distance along volcplane from volcano to image centre
d_volcplaneshort = d_volcplane * np.sin(abs(a_volc))

#Calculate angle between plume direction and implane
a_alpha = (np.pi/2) + dir_imcentre - dir_plume

#Ensure alpha is between -90 and +90 degrees
while a_alpha < -np.pi/2 or a_alpha > np.pi/2:
    if a_alpha <= -np.pi/2:
        a_alpha += np.pi
    elif a_alpha >= np.pi/2:
        a_alpha -= np.pi
        
#Calculate distance to image plane so centre of image plane intersects plume
d_implane = d_volcplane + d_volcplaneshort * np.tan(a_alpha)

# Calculate angle between inverse plume direction and viewing direction
theta = np.pi/2 - a_alpha

#Create map of pixels with plus and minus
x_grid = np.arange(-res_x/2,res_x/2)
y_grid = np.arange(-res_y/2,res_y/2)

# Create pixel coordinates for image
im_coord = np.meshgrid(x_grid,-y_grid)

px_angle = im_coord[0] * fov/res_x
dist_correction = (np.cos(px_angle)+np.divide((np.sin(px_angle)),np.tan(theta-px_angle)))

dist = d_implane * dist_correction

#np.save('F:/Black_Hole/Data/201801_Masaya/20180110/so2_camera/measurements/plume_distance/612-866_310.npy',dist)
