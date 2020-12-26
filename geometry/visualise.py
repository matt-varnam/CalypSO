# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:09:58 2020

@author: Matthew Varnam - The University of Manchester
@email: matthew.varnam(-at-)manchester.ac.uk
"""
#Import numpy for mathematical calculations
import numpy as np

#Import shapely to test speed compared to normal algorithm  
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

#Import multiprocessing to exploit multiple CPU cores for calculation
from multiprocessing import Pool, cpu_count

#Import tqdm to monitor progression of multiprocessing Pool
import tqdm

#Import custom module classes, functions and multiprocessing worker code
from geocamp.geometry.module import Camera, Topography
from geocamp.geometry.quick_pp import worker_qp, proc_qp

def model_terrain(virtual_cam,masaya_topography):

    #Flatten terrain onto image plane
    cam_xyz , tot_dist   = virtual_cam.cam_3Dcartesian(masaya_topography)
    (px, py), cam_coords = virtual_cam.cam_2Dcartesian(cam_xyz)
   
    #Create empty arrays to store final polygons and distance to each polygon
    shape = np.shape(px)
    polygons      = np.empty((shape[0]-1,shape[1]-1,2,3,2))
    mean_distance = np.empty((shape[0]-1,shape[1]-1,2))
    
    #------------------------------------------------------------------------------
    #-----  Create polygons to plot image
    
    print('Triangularising terrain')
    
    #Create and store the polygons
    for j in np.arange(shape[1]-1):
        for i in np.arange(shape[0]-1):
            for k in np.arange(2):
                if k == 0:
                    #Create single polygon corners
                    polygon = np.array([[px[i,j]    ,py[i,j]],
                                        [px[i+1,j]  ,py[i+1,j]],
                                        [px[i,j+1]  ,py[i,j+1]]])
                    
                    vertex_dist = np.array([tot_dist[i  ,j  ],
                                            tot_dist[i+1,j  ],
                                            tot_dist[i  ,j+1]])
                    
                elif k == 1:
                    #Create second polygon corners
                    polygon = np.array([[px[i+1,j]  ,py[i+1,j]],
                                        [px[i+1,j+1],py[i+1,j+1]],
                                        [px[i,j+1]  ,py[i,j+1]]])
                    
        
                    vertex_dist = np.array([tot_dist[i+1,j  ],
                                            tot_dist[i+1,j+1],
                                            tot_dist[i  ,j+1]])
                    
                poly_distance = np.mean(vertex_dist)
                    
                mean_distance[i,j,k] = poly_distance               
                polygons[i,j,k]       = polygon
    
    #Cut polygon grid to only points the camera can see
    sq_cam_check = cam_coords[2][:-1,:-1]
    
    #Double arrays due to triangles
    tri_cam_check = np.dstack((sq_cam_check,sq_cam_check)) 
    double_x = np.dstack((px[:-1,:-1],px[:-1,:-1]))
    double_y = np.dstack((py[:-1,:-1],py[:-1,:-1]))
    
    #Select all pixels in the field of view of the camera within 6 km
    logic0 = np.logical_and(double_x >= -200,
                            double_x < virtual_cam.res_x + 200)
    
    logic1 = np.logical_and(double_y >= -200,
                            double_y < virtual_cam.res_y + 200)
    
    logic2 = np.logical_and(tri_cam_check < 0,
                            mean_distance < virtual_cam.render_distance)
    
    #Combine logic
    logic  = np.logical_and(np.logical_and(logic0,logic1),logic2)
                       
    #Select polygons and distances based on above logic    
    reshaped_polygons = polygons[logic]
    distance_km = (mean_distance[logic]/1000)
    
    #Sort the polygons by distance to the camera
    sort = np.flip(np.argsort(distance_km))
    sorted_distance = distance_km[sort]
    sorted_polygons = reshaped_polygons[sort]

    #Create points
    im_x = np.arange(0,virtual_cam.res_x)
    im_y = np.arange(0,virtual_cam.res_y)  
    px_points = np.dstack(np.meshgrid(im_x,im_y))
    px_points = px_points.reshape(np.multiply(virtual_cam.res_y,
                                              virtual_cam.res_x)
                                  ,2) 
    
    return sorted_polygons,sorted_distance,px_points

def create_hillside(sorted_polygons,px_points,virtual_cam,cores = None):
    
    im_x = np.arange(0,virtual_cam.res_x)
    
    print('Creating hillside boolean mask')
    #Create shapely geometry polygons
    polygon_list = [Polygon(poly) for poly in sorted_polygons]
    
    #Combine polygons to single polygon with a small buffer to ensure success
    merged_poly = unary_union([geom.buffer(0.01) for geom in polygon_list 
                               if geom.is_valid])
    
    #Convert shapely object to xy coordinates
    poly_exterior = merged_poly.exterior.xy
    
    #Convert input arguments to lists to allow pickling
    outline_list  = list(poly_exterior)
    px_point_list = px_points.tolist()
    resolution    = (virtual_cam.res_x,virtual_cam.res_y)
    
    #Put all arguments together to be given to initializer worker
    args = px_point_list, outline_list, resolution
    
    #Take out of numpy array form
    pq_params = im_x.tolist()
    
    #Create multiprocessing task pool to see if points lie in single polygon
    with Pool(processes = cores, initializer = worker_qp,initargs = [args,]) as p0:
        mapped_values = list(tqdm.tqdm(p0.imap_unordered(proc_qp,pq_params), 
                                       total = len(pq_params)))
    
    #Create boolean map to store final hillside
    hill_bool = np.full((virtual_cam.res_y,virtual_cam.res_x),0,dtype = 'bool')
               
    #Create boolean mask from hillside boundary
    for i,pt in mapped_values:
        if i > 0:

            hill_bool[pt:,i] = 1
        
    return hill_bool