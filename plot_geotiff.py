# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:11:02 2020

Plot a geotif image and attempt to extract how far the camera is from the 
volcano for the entire camera's field of view

@author: Matthew Varnam - The University of Manchester
@email: matthew.varnam(-at-)manchester.ac.uk
"""
#------------------------------------------------------------------------------
#-----  Import external python libraries

#Import matplotlib for plotting functions
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Import numpy for mathematical calculations
import numpy as np

#Import leastsq for optimising virtual camera rotations
from scipy.optimize import minimize

#Import shapely to test speed compared to normal algorithm  
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
from shapely.ops import unary_union

#Import library for reading 16-bit images
import cv2

#Import multiprocessing to exploit multiple CPU cores for calculation
from multiprocessing import Pool

#Import tqdm to monitor progression of multiprocessing Pool
import tqdm

#Import custom module classes, functions and multiprocessing worker code
from geocamp.geometry.module import Camera, Topography
from geocamp.geometry.module import load_topography,haversine_ang
from geocamp.geometry.point_in_poly import worker_pp,proc_pp
from geocamp.geometry.visualise import model_terrain, create_hillside
from geocamp.geometry.quick_pp import worker_qp, proc_qp

#Import custom SO2 camera module for image offset detection
from CalypSO.image_registration import detectshift

from glob import glob

#Create function to calculate residual between the two images
def residual(fit_params, *args):
    
    virtual_cam, topography, cam_intensity = args
        
    #Update camera viewing direction
    virtual_cam.view_azimuth   = fit_params[0]
    virtual_cam.view_elevation = fit_params[1]
        
    polygons,distance,cam_coords = model_terrain(virtual_cam,topography)
    
    #Create hillside mask from topography outputs
    hill_bool = create_hillside(polygons,cam_coords, virtual_cam)
    hill_line = hill_bool.astype('uint8') * 1000
    
    #Create array representing zero image offset
    no_shift = np.array([[1,0,0],[0,1,0]])
    
    try:
        #Detect shift between real and virtual hillsides
        shift = detectshift(cam_intensity,hill_line,kernal = 31,mode=cv2.MOTION_TRANSLATION)
    
        #Find the difference between the measured shift and zero shift
        error = np.subtract(shift, no_shift).ravel()
        
        residual = np.sum(np.square(error))
    except cv2.error:
        residual = 100000
    
    print(virtual_cam.view_azimuth,
          virtual_cam.view_elevation, 
          virtual_cam.view_roll,
          residual)
    
    return residual

def minimise_hillside(virtual_cam,topography,reference_image):
    
    #Use initial values of azimuth and elevation for fitting
    fit_params = (virtual_cam.view_azimuth,virtual_cam.view_elevation)
    
    #Define bounds for each angle
    ang_lim = np.array([[fit_params[0] - 2, fit_params[0] + 2],
                        [fit_params[1] - 2, fit_params[1] + 2]])
    
    #Create COBYLA constrains from bounds
    cons = []
    for factor in range(len(ang_lim)):
        lower, upper = ang_lim[factor]
        l = {'type': 'ineq',
             'fun': lambda x, lb=lower, i=factor: x[i] - lb}
        u = {'type': 'ineq',
             'fun': lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(l)
        cons.append(u)
        
    #Minimize residual created by image offset
    fit_results = minimize(residual,fit_params,
                          args = (virtual_cam, topography, reference_image),
                          method='COBYLA',
                          constraints = cons)
        
    #Output azimuth and elevation
    virtual_cam.view_azimuth   = fit_results['x'][0]
    virtual_cam.view_elevation = fit_results['x'][1]
    
    #Print solution
    print('Solution: ' + 
          str(virtual_cam.view_azimuth)   + ' ' +
          str(virtual_cam.view_elevation) + ' ' +
          str(virtual_cam.view_roll) + ' ' )
    
    return virtual_cam

def render(virtual_cam, topography):
    

    #Create output array
    pixel_distance = np.full((virtual_cam.res_y,
                              virtual_cam.res_x),-1,dtype = 'float64')
    
    sorted_polygons,sorted_distance,px_points = \
        model_terrain(virtual_cam,topography)
        
    print('Render terrain')
    num_tasks = len(sorted_polygons)
        
    px_point_list = px_points.tolist()
    sorted_polygon_list = sorted_polygons.tolist()
    args = [px_point_list,sorted_polygon_list]
    
    pp_params = np.arange(num_tasks).tolist()
    
    with Pool(initializer = worker_pp,initargs = [args,]) as p1:
        mapped_values = list(tqdm.tqdm(p1.imap_unordered(proc_pp, pp_params), 
                                       total = num_tasks))
        
    for k,px_coords in mapped_values:
        for j,i in px_coords:
            if pixel_distance[i,j] > 0:
                if pixel_distance[i,j] > sorted_distance[k]:
                    pixel_distance[i,j] = sorted_distance[k]
                    
            else:
                pixel_distance[i,j] = sorted_distance[k]
    
    # #Section commented out is an alternative single-core method for tracing
    # #Create shapely point list
    # point_list   = [Point(point)  for point in px_points]

    # #Create STRtree from points with recorded ids
    # tree = STRtree(point_list)
    # index_by_id = dict((id(pt), i) for i, pt in enumerate(point_list))
    
    # #Loop over polygons
    # for k,poly in enumerate(tqdm.tqdm(polygon_list,total = num_tasks)):
        
    #     #print(str(k) + ' / ' + str(num_tasks))
        
    #     valid_ids = [(index_by_id[id(pt)]) for pt in tree.query(poly)
    #                     if pt.intersects(poly)]
        
    #     valid_points = px_points[valid_ids]
        
    #     for j,i in valid_points:
    #         if pixel_distance[i,j] > 0:
    #             if pixel_distance[i,j] > sorted_distance[k]:
    #                 pixel_distance[i,j] = sorted_distance[k]
    
    #         else:
    #             pixel_distance[i,j] = sorted_distance[k]
               
    #Import PolyCollection to allow plotting of polygons
    from matplotlib.collections import PolyCollection
    
    #Create single object of polygons to be plotted with colorscheme attached
    p = PolyCollection(sorted_polygons,array = sorted_distance)
    
    #Create axes for plot
    fig, ax = plt.subplots()
    
    #Add polygons to axes
    im = ax.add_collection(p)
    
    #Adjust plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0,1600)
    plt.ylim(1200,0)
    plt.colorbar(p,label='Distance (km)')
    plt.axis('off')
    plt.show()
    
    return pixel_distance
    
def master():
    #--------------------------------------------------------------------------
    #-----  Script setup with preset values
    
    #Define other properties
    render_dist = 6000
    lat_min = 11.96
    lat_max = 12.04
    lon_min = -86.22
    lon_max = -86.12
    
    #Define camera properties
    res_x = 1600
    res_y = 1200
    fov_x = 26.5
    view_az = 176.23377824742465
    view_el = 5.355369647143472
    view_rl = 5.445438414812088

    cam_lat = 12.02446
    cam_lon = -86.17946
    cam_hgt = 320

    #-------------------------------------------------------------------------
    #-----  Load topography from pre-downloaded NASA data
    
    print('Load Topography')
    
    #Create master directory
    topo_dir = 'C:/COVID/SRTM_Masaya/'
    
    #Load topography data field 1
    topo_fpath0 = topo_dir + 'n12_w087_1arc_v3.tif'
    masaya_topography = load_topography(topo_fpath0)
    
    #Load topography data field 2
    topo_fpath1 = topo_dir + 'n11_w087_1arc_v3.tif'
    south_topo = load_topography(topo_fpath1)
    
    #Combine the topography data from the two geotifs
    masaya_topography.add(south_topo,'d')
    
    #Trim topography
    masaya_topography.trim(lat_min,lat_max,lon_min,lon_max)
    
    #-------------------------------------------------------------------------
    #-----  Load real images to virtually orientate the camera
        
    #Set location of camera imagery
    cam_dir = 'F:/Black_Hole/Data/201801_Masaya/20180113/'
    cam_dir = (cam_dir + 'Camera0/')
    
    cam_fpath = glob(cam_dir + '*')[20]
    
    #Load camera imagery
    cam_int = cv2.imread(cam_fpath,-1)
    
    #Rotate and flip camera image to match real world orientation
    cam_int = np.rot90(cam_int,1)
    cam_int = np.fliplr(cam_int)
    
    #-------------------------------------------------------------------------
    #-----  Convert from 3d real world coordinates to camera coordinates
        
    #Summarise properties
    cam_loc  = (cam_lat,cam_lon,cam_hgt)
    cam_view = (view_az,view_el,view_rl)
    cam_prop = (res_x,res_y,fov_x)
    
    #Create camera
    qsi_cam = Camera(cam_loc,cam_view,cam_prop,render_dist)
        
    #Create polygons for the terrain    
    polygons,distance,cam_coords = model_terrain(qsi_cam,masaya_topography)
    
    #Create hillside mask from topography outputs
    hill_bool = create_hillside(polygons,cam_coords, qsi_cam,cores = 1)
    hill_line = hill_bool.astype('uint8') * 1000
    
    #Detect offset between virtual and real terrain
    shift = detectshift(cam_int,hill_line, graphing = True, kernal = 31)
        
    #Convert roll required to rotation angle needed
    rl_rot_rad0 = np.arccos(shift[0][0])
    rl_rot_rad1 = -np.arcsin(shift[0][1])
    
    if  rl_rot_rad0 < 0 and rl_rot_rad1 > 0 or rl_rot_rad0 > 0 and rl_rot_rad1 < 0:
        rl_rot = -np.degrees(rl_rot_rad0)
    
    else:
        rl_rot = np.degrees(rl_rot_rad0)
    
    #Update camera viewing direction
    qsi_cam.view_roll = qsi_cam.view_roll + rl_rot
    
    #Print current values
    print(qsi_cam.view_azimuth,qsi_cam.view_elevation,qsi_cam.view_roll)
    
    #Ask user if initial fit is good enough
    cont = input('Iterate better guess?- y or n:')
    
    if cont == 'y' or cont == 'Y':
        qsi_cam = minimise_hillside(qsi_cam,masaya_topography,cam_int)
    
    #Ask user to produce final hillside map
    cont = input('Render solution?- y or n:')

    if cont == 'y' or cont == 'Y':
        pixel_distance = render(qsi_cam,masaya_topography)
        
        fig,ax = plt.subplots(1,1)
        im = ax.imshow(pixel_distance)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05) 
        ax.axis('off')   
        fig.colorbar(im, cax = cax, label = 'Distance (km)')
        plt.show()
        sav_dir = 'F:/Black_Hole/Data/201801_Masaya/20180113/measurements/hill_distance/1-186_310_2.npy'
        np.save(sav_dir,pixel_distance)

if __name__ == '__main__':
    master()
'''    
    #------------------------------------------------------------------------------
    #-----  Script setup with preset values
    
    #Define other properties
    render_dist = 6000
    lat_min = 11.96
    lat_max = 12.04
    lon_min = -86.22
    lon_max = -86.12
    
    #Define camera properties
    res_x = 1600
    res_y = 1200
    fov_x = 26.5
    view_az = 170 #170.00 #170.75
    view_el = 8.0   #8.1 #7.64
    view_rl = 0   #0.8 #0.7
    cam_lat = 12.02446
    cam_lon = -86.17946
    cam_hgt = 320
    
    #multicore = input('Run with multiple CPU cores - y or n:')
    
    #-------------------------------------------------------------------------
    #-----  Load topography from pre-downloaded NASA data
    
    print('Load Topography')
    
    #Create master directory
    topo_dir = 'C:/COVID/SRTM_Masaya/'
    
    #Load topography data field 1
    topo_fpath0 = topo_dir + 'n12_w087_1arc_v3.tif'
    masaya_topography = load_topography(topo_fpath0)
    
    #Load topography data field 2
    topo_fpath1 = topo_dir + 'n11_w087_1arc_v3.tif'
    south_topo = load_topography(topo_fpath1)
    
    #Combine the topography data from the two geotifs
    masaya_topography.add(south_topo,'d')
    
    #Trim topography
    masaya_topography.trim(lat_min,lat_max,lon_min,lon_max)
    
    #-------------------------------------------------------------------------
    #-----  Convert from 3d real world coordinates to camera coordinates
    
    #leastsq(func,x0,args = (),full_output = True, **kwargs))
    
    print('Convert to image coordinates')
    
    #Summarise properties
    cam_loc  = (cam_lat,cam_lon,cam_hgt)
    cam_view = (view_az,view_el,view_rl)
    cam_prop = (res_x,res_y,fov_x)
    
    qsi_cam = Camera(cam_loc,cam_view,cam_prop)
    
    cam_xyz,tot_dist = qsi_cam.cam_3Dcartesian(masaya_topography)
    
    (px, py),cam_coords = qsi_cam.cam_2Dcartesian(cam_xyz)
    
    print('Trim topography grid')
    
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
    
    #Double array due to triangles
    tri_cam_check = np.dstack((sq_cam_check,sq_cam_check))
    
    double_x = np.dstack((px[:-1,:-1],px[:-1,:-1]))
    double_y = np.dstack((py[:-1,:-1],py[:-1,:-1]))
    
    #Select all pixels in the field of view of the camera within 6 km
    logic0 = np.logical_and(double_x >= -200,double_x < res_x + 200)
    logic1 = np.logical_and(double_y >= -200,double_y < res_y + 200)
    
    logic2 = np.logical_and(tri_cam_check < 0,
                            mean_distance < render_dist)
    
    logic  = np.logical_and(np.logical_and(logic0,logic1),logic2)
                           
    reshaped_polygons = polygons[logic]
    distance_km = (mean_distance[logic]/1000)
    
    #Sort the polygons by distance to the camera
    sort = np.flip(np.argsort(distance_km))
    sorted_distance = distance_km[sort]
    sorted_polygons = reshaped_polygons[sort]
    
    #Create points
    im_x = np.arange(0,res_x)
    im_y = np.arange(0,res_y)  
    px_points = np.dstack(np.meshgrid(im_x,im_y)).reshape(res_y*res_x,2) 
    
    print('Converting to camera pixelwise view')
        
    ### -Add section to match polygons to hillside without full calculation
    #Create shapely geometry objects
    point_list   = [Point(point)  for point in px_points]
    polygon_list = [Polygon(poly) for poly  in sorted_polygons]
    
    # code to scan through each column for intersection, and if found, set all
    #pixels below as true
    
    merged_poly = unary_union([geom.buffer(0.01) for geom in polygon_list 
                               if geom.is_valid])
    
    #Convert shapely object to xy coordinates
    poly_exterior = merged_poly.exterior.xy

    #Convert input arguments to lists to allow pickling
    outline_list  = list(poly_exterior)
    px_point_list = px_points.tolist()
    resolution    = (res_x,res_y)
    
    #Put all arguments together to be given to initializer worker
    args = px_point_list, outline_list, resolution
    
    #Take out of numpy array form
    pq_params = im_x.tolist()
    
    #Create multiprocessing task pool
    with Pool(initializer = worker_qp,initargs = [args,]) as p0:
        mapped_values = list(tqdm.tqdm(p0.imap_unordered(proc_qp,pq_params), 
                                       total = len(pq_params)))
        
        sorted(mapped_values)[1]
        
    #Create boolean map to store final hillside
    hill_bool = np.full((res_y,res_x),0,dtype = 'bool')
        
    hill_bol = hill_bool.astype('uint8') * 1000
'''