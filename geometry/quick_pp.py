# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:02:37 2020

A faster multiprocessing solution for the point in polygon problem. This code
assumes that when moving down an image, once a hillside has been found, all 
pixels below that point will also be hillside.

@author: Matthew Varnam - The University of Manchester
@email: matthew.varnam(-at-)manchester.ac.uk
"""

#Import numpy for mathematical calculations
import numpy as np

#Import shapely to create easily queryable objects
from shapely.geometry import Point, Polygon

#Create controlling worker to operate multiple procedures
def worker_qp (point_poly_res):
    
    #Create global variables to be used by the procedures
    global point_list
    global merged_poly
    global res_x
    global res_y
    
    #Extract points and polygon lists from initialiser function argument
    point_list, poly_list, resolution = point_poly_res
    
    #Extract x and y resolutions from tuple
    res_x,res_y = resolution
    
    px_points = np.array(point_list)
    poly_array = np.array(poly_list).T
    
    point_list  = [Point(point) for point in px_points]
    merged_poly = Polygon(poly_array)

#Procedure function to be called multiple times to identify points in polygons
def proc_qp (*args):
    j = args[0]
    horizon = False
    i = 0
    
    returner = -1
    
    while horizon == False and i < res_y:
        p_value = i * res_x + j
        query_point = point_list[p_value]
        
        if merged_poly.contains(query_point):
            returner = i
            horizon = True
            
        else:
            i += 1
     
    return j,returner