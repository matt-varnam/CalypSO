# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:10:46 2020

Script providing code to rotate 3D cartesian vectors to point in new directions

@author: Matthew Varnam - The University of Manchester
@email: matthew.varnam(-at-)manchester.ac.uk
"""

#Import numpy for mathematical calculations
import numpy as np

#Create yaw, pitch and roll matrices for transformations
def yaw(alpha):
    '''
    Short function to rotate cartesian coordinates clockwise around z axis
    
    Parameters
    ----------
    alpha : FLOAT
        A float for the number of clockwise degrees to rotate by.

    Returns
    -------
    matrix : ARRAY
        Gives the rotation matrix required to rotate the spherical coordinates
    '''
    
    alpha = np.radians(alpha)
    matrix = np.array(([np.cos(alpha) ,-np.sin(alpha),0],
                       [np.sin(alpha) , np.cos(alpha),0],
                       [0             ,       0      ,1]))
    return matrix

def pitch(beta): 
    '''
    Short function to rotate cartesian coordinates clockwise around y axis
    
    Parameters
    ----------
    beta : FLOAT
        A float for the number of clockwise degrees to rotate by.

    Returns
    -------
    matrix : ARRAY
        Gives the rotation matrix required to rotate the spherical coordinates
    '''
    
    beta = np.radians(beta)
    matrix = np.array(([np.cos(beta) ,0, np.sin(beta)],
                       [0            ,1,            0],
                       [-np.sin(beta),0, np.cos(beta)]))
    return matrix

def roll(gamma):
    '''
    Short function to rotate cartesian coordinates clockwise around x axis
    
    Parameters
    ----------
    gamma : FLOAT
        A float for the number of clockwise degrees to rotate by.

    Returns
    -------
    matrix : ARRAY
        Gives the rotation matrix required to rotate the spherical coordinates
    '''
    gamma = np.radians(gamma)
    matrix = np.array(([1 ,     0        ,             0],
                       [0 , np.cos(gamma),-np.sin(gamma)],
                       [0 , np.sin(gamma), np.cos(gamma)]))
    return matrix