# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:41:30 2020

@author: Matthew Varnam - The University of Manchester
@email: matthew.varnam(-at-)manchester.ac.uk
"""

#Library to open geotiff images
from osgeo import gdal

#Import numpy for mathematical calculations
import numpy as np

#Import rotation functions
from geocamp.geometry.rotation import roll, pitch, yaw

#------------------------------------------------------------------------------
#-----  Define new functions

def load_topography(topo_fpath):
    '''
    Load topography from specified filepath and store it in a specific
    Topography class

    Parameters
    ----------
    topo_fpath : String
        The full filepath to pre-downloaded NASA imagery of the area desired.
        Only tested for 1arc_v3 .tif imagery

    Returns
    -------
    topo : Topography
        Topography object containing loaded latitude,longitude and altitude

    '''
    gdal_data = gdal.Open(topo_fpath,gdal.GA_ReadOnly)
    gt = gdal_data.GetGeoTransform()
    raster_band = gdal_data.GetRasterBand(1)
    altitude = raster_band.ReadAsArray()
    
    #Extract shape of raster
    gdal_shape = altitude.shape
    
    #Extract affine data
    top_left_lon, width, rot0, top_left_lat, rot1, height = gt

    #Code currently does not add rotation here. To add later
    if rot0 != 0.0 or rot1 != 0.0:
        raise Exception('Code cannot currently use topography that is \
                        aligned north-up orientated')
                        
    #Create pixel indices labels for x and y axes
    px_ygrid = np.arange(0,gdal_shape[0],1)
    px_xgrid = np.arange(0,gdal_shape[1],1)
    
    #Create lat and lon grid
    lat_grid = np.multiply(px_ygrid,height) + top_left_lat
    lon_grid = np.multiply(px_xgrid,width)  + top_left_lon
    
    #Create full x,y grid for raster
    lonlat_grid = np.meshgrid(lon_grid,lat_grid)
    
    #Add data to topography class
    topo = Topography(lonlat_grid[1],lonlat_grid[0],altitude)
    
    return topo

def haversine_ang(lon1, lat1, lon2, lat2):
    '''
    Function to calculate the displacement and bearing between two 
    GPS coordinates
    
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
    
    # Calculate the bearing
    bearing = np.arctan2(np.sin(dlon) * np.cos(lat2), 
                         np.cos(lat1) * np.sin(lat2) - 
                         np.sin(lat1) * np.cos(lat2) * np.cos(dlat))
                    
    # Convert barings to the range (0, 2pi) instead of (-pi, pi)
    bearing = np.where(bearing < 0,2 * np.pi + bearing,bearing)
    
    return c, bearing

#------------------------------------------------------------------------------
#-----  Define new classes

#Create a class for the specific camera location
class Camera:
    
    def __init__(self, location, view, properties, render_distance):
        '''
        Create a Camera class to place onto topography, define its viewing
        direction then 'take a picture' using the camera's properties

        Parameters
        ----------
        location : TUPLE
            tuple containing latitude,longitude and altitude
        view : TUPLE
            tuple containing view_azimuth, view_elevation and view_roll
        properties : TUPLE
            tuple containing x and y resolution of the camera, as well as the
            field of view in the x direction
        render_distance : INT
            distance to render terrain to in metres

        '''
        
        #Setup Camera class location
        self.latitude = location[0]
        self.longitude = location[1]
        self.altitude = location[2]
        
        #Setup Camera class viewing orientation and direction
        self.view_azimuth   = view[0]
        self.view_elevation = view[1]
        self.view_roll      = view[2]
        
        #Setup Camera class camera properties
        self.res_x   = properties[0]
        self.res_y   = properties[1]
        self.field_of_view = properties[2]
        
        #Set Camera class render distance
        self.render_distance = render_distance
        
    def cam_3Dcartesian(self,topography):
        '''
        Convert Topography class object to 3D cartesian coordinates with the
        camera located at (0,0,0)
        
        Parameters
        ----------
        topography : Topography
            A Topography class object containing a latitude, longitude and 
            altitude 3D grids

        Returns
        -------
        land_cam : ARRAY
            Array containing 3 2D arrays with x, y and z coordinates of each
            topography point relative to the camera.
        tot_dist : ARRAY
            A 2D .

        '''
        
        #Calculate angular distance from camera to each point in terrain
        ang_dist, bearing = haversine_ang(self.longitude,self.latitude,
                                          topography.longitude,
                                          topography.latitude)
        
        #Calculate total distance
        e_radius = 6371E3
        flat_dist = np.multiply(ang_dist, e_radius)
        
        #Calculate earth fallaway due to straight line on a sphere
        falloff = np.divide(e_radius , np.cos(ang_dist)) - e_radius
        
        #Use falloff to calculate total relative elevation difference
        relative_altitude = (topography.altitude - self.altitude) - falloff
        
        #Calculate elevation angle to ground
        relative_elevation = np.arctan(np.divide(relative_altitude,flat_dist))
        
        #Calculate the total distance from camera to ground
        tot_dist = np.sqrt(np.power(flat_dist,2) + 
                           np.power(relative_altitude,2))
        
        #Convert angles to sperical coordinate system
        theta = - relative_elevation + (np.pi/2)
        phi = bearing - (np.pi/2)
        phi = np.where(phi < 0, phi + 2*np.pi,phi)
        phi = 2*np.pi - phi
        
        #Convert to camera cartesian coordinates
        x_land_cam = np.multiply(np.multiply(tot_dist,np.sin(theta)),
                                 np.cos(phi))
        y_land_cam = np.multiply(np.multiply(tot_dist,np.sin(theta)),
                                 np.sin(phi))
        z_land_cam = np.multiply(tot_dist,np.cos(theta))
        
        #Create array containing x, y and z for each point relative to camera
        xyz_cam = np.array((x_land_cam,y_land_cam,z_land_cam))
        
        return xyz_cam,tot_dist
    
    def cam_2Dcartesian(self,land_cam):
        
        #Rotate cartesian so looking at z = -1
        r0 = 360 - self.view_azimuth  
        r1 = self.view_roll
        r2 = 90 + self.view_elevation
        
        R = np.dot(np.dot(roll(-r2),pitch(-r1)),yaw(-r0))
        
        cam_coords = np.dot(land_cam.T, R.T).T
        
        bx = np.multiply((-1/cam_coords[2]),cam_coords[0])
        by = np.multiply((-1/cam_coords[2]),cam_coords[1])
        
        bx = np.divide(bx,np.tan(np.radians(self.field_of_view/2)))
        by = np.divide(by,np.tan(np.radians(self.field_of_view/2)))
        
        #Include aspect ratio
        aspect_ratio = np.divide(self.res_x,self.res_y)
        
        #Recentre pixels so 0,0 is top-left corner and 1,1 is bottom right
        perx = np.divide(bx + 1,2)
        pery = np.divide(1 - np.multiply(by,aspect_ratio),2)
        
        #Convert to pixel number so bottom right is 1600,1200
        px = np.multiply(perx,self.res_x)-0.5
        py = np.multiply(pery,self.res_y)-0.5
        
        return (px,py),cam_coords
        
#Create a class to hold the topography
class Topography:
    
    def __init__(self, lat_grid, lon_grid, altitude):
        '''
        The Topography class stores the latitude, longitude and altitude, as
        well as having a method to combine two topographies
        
        Parameters
        ----------
        lat_grid : ARRAY
            3D Array with latitude measurements of each altitude point
        lon_grid : ARRAY
            3D Array with longitude measurements of each altitude point
        altitude : ARRAY
            3D Array with altitude measurements of topography

        '''
        #Setup grid
        self.latitude  = lat_grid
        self.longitude = lon_grid
        self.altitude  = altitude
    
    def add(self, topo, direction):
        '''
        Add a new topography grid to the topography class

        Parameters
        ----------
        topo : Topography
            The new topography to be added to the Topography class
        direction : STRING
            define as 'u','d','l','r' for the direction the new bit of 
            topography is being added in

        '''
        if direction == 'u':
            self.latitude  = np.vstack((topo.latitude, self.latitude))
            self.longitude = np.vstack((topo.longitude,self.longitude))
            self.altitude  = np.vstack((topo.altitude, self.altitude))
        
        elif direction == 'd':
            self.latitude  = np.vstack((self.latitude, topo.latitude))
            self.longitude = np.vstack((self.longitude,topo.longitude))
            self.altitude  = np.vstack((self.altitude ,topo.altitude))
            
        elif direction == 'l':
            self.latitude  = np.hstack((topo.latitude, self.latitude))
            self.longitude = np.hstack((topo.longitude,self.longitude))
            self.altitude  = np.hstack((topo.altitude, self.altitude))
            
        elif direction == 'r':
            self.latitude  = np.hstack((self.latitude, topo.latitude))
            self.longitude = np.hstack((self.longitude,topo.longitude))
            self.altitude  = np.hstack((self.altitude ,topo.altitude))
            
    def trim(self,lat_min,lat_max,lon_min,lon_max):
        
        #Create 
        logic_lat = np.logical_and(self.latitude > lat_min,
                                   self.latitude < lat_max)
        
        logic_lon = np.logical_and(self.longitude > lon_min,
                                   self.longitude < lon_max)
        
        lat_limit = [np.argwhere(logic_lat)[i][0] for i in [0,-1]]
        lon_limit = [np.argwhere(logic_lon)[i][1] for i in [0,-1]]
        
        #Trim longitude and latitude
        self.latitude  = self.latitude[lat_limit[0]:lat_limit[1],
                                       lon_limit[0]:lon_limit[1]]
        self.longitude = self.longitude[lat_limit[0]:lat_limit[1],
                                       lon_limit[0]:lon_limit[1]]
        self.altitude  = self.altitude[lat_limit[0]:lat_limit[1],
                                       lon_limit[0]:lon_limit[1]]