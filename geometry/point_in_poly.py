# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:10:42 2020

Small module to allow multiprocessing of the point in polygon problem

@author: Matthew Varnam - The University of Manchester
@email: matthew.varnam(-at-)manchester.ac.uk
"""

#Import numpy for mathematical calculations
import numpy as np

#Import shapely to create easily queryable objects
from shapely.geometry import Point, Polygon
from shapely.strtree  import STRtree

#Create controlling worker to operate multiple procedures
def worker_pp (point_poly_list):

    #Create global variables to be used by the procedures
    global tree
    global polygon_list
    global px_points
    global index_by_id
    
    #Extract points and polygon lists from initialiser function argument
    point_list,poly_list = point_poly_list
    px_points = np.array(point_list)
    sorted_polygons = np.array(poly_list)
    
    #Convert to shapely class Points and Polygons
    point_list   = [Point(point) for point in px_points]
    polygon_list = [Polygon(poly) for poly  in sorted_polygons]
    
    #Create STRtree to speed up checking of points and polygons
    tree = STRtree(point_list)

    #Create dictionary to index point list for faster querying
    index_by_id = dict((id(pt), i) for i, pt in enumerate(point_list))

#Procedure function to be called multiple times to identify points in polygons
def proc_pp (*args):
    
    #Choose the polygon matching the index provided by the multiprocessing Pool
    k = args[0]   
    poly = polygon_list[k]
    
    #Conduct two things - query the STRtree then confirm an intersection
    valid_ids = [(index_by_id[id(pt)]) for pt in tree.query(poly)
                    if pt.intersects(poly)]
    
    #Find the coordinates of the points that lie inside the polygon
    valid_points = px_points[valid_ids]
    returner = valid_points.tolist()
     
    return k,returner