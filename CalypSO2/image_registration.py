# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:18:10 2017

This module is used for image registration for SO2 camera images. It takes two
 images, works out the offset between the two, then can be used to shift one
                        image to eliminate the offset

@author: Matthew Varnam - The University of Manchester
@email: matthew.varnam(-at-)manchester.ac.uk
"""
##CV2 is an image processing library used to import tif images
import cv2

##Numpy is the primary library for array (matrix) processing, since it is 
#significantly faster than base Python through the use of C programming
import numpy as np

#Import image conversion package from CalypSO
from CalypSO.change_datatype import uint16_uint8

##Matplotlib provides plotting functions for displaying the images produced
import matplotlib.pyplot as plt

##Function detects the offset between two images and caculates the matrix
#required to map the second image on to the first

#Mode is either cv2.MOTION_TRANSLATION or cv2.MOTION_EUCLIDEAN
#Termination_eps specifies the threshold
## of the increment in the correlation coefficient between two iterations
#Iterations is the number of iterations undertaken
def detectshift(image_310,image_330,mode = cv2.MOTION_EUCLIDEAN,
                iterations = 5000, termination_eps = 1e-7,graphing = False,
                kernal = 3):

    #Convert raw images into gradient images for edge finding
    image0 = get_gradient(image_310,kernal = kernal)
    image1 = get_gradient(image_330,kernal = kernal)    
    
    #image0 = uint16_uint8(image_310)
    #image1 = uint16_uint8(image_330)
    
    #Plot images before alignment
    if graphing == True:
        check_alignment(image0,image1)
    
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    warp_matrix = np.eye(2, 3, dtype=np.float32)
 
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations,  termination_eps)
 
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (image0,image1,warp_matrix, mode, criteria)
    
    #Plot images after alignment
    if graphing == True:
        check_alignment(image0,image1,warp_matrix)
    
    return warp_matrix
    
##Function takes second image from detectshift and performs the transformation.
def alignment(image,warp_matrix):
    # Use warpAffine for Translation, Euclidean and Affine
    im_aligned = cv2.warpAffine(image, warp_matrix,
                                (image.shape[1],image.shape[0]),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    return im_aligned

#Calculate the gradient of an image using a kernal size given by ksize    
def get_gradient(im,kernal = 3):
    convert = im.astype('float32')
    
    im = cv2.blur(im,(5,5))
    
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(convert,cv2.CV_64F,1,0,ksize = kernal)
    grad_y = cv2.Sobel(convert,cv2.CV_64F,0,1,ksize = kernal)
    
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    grad = cv2.blur(grad,(5,5))
    grad2 = (grad * (254/np.max(grad)))
    grad2 = grad2.astype('uint8')

    return grad2
    
#Function takes two image gradients and plots them on top of each other
def grad_im(gradim0,gradim1):
    #gradim 0 is colour channel 1
    #gradim 1 is colour channel 2
    #gradim 2 is colour channel 3
    gradim0 = uint16_uint8(gradim0)
    gradim1 = uint16_uint8(gradim1)
    
    #Blue channel is blank
    gradim2 = np.zeros_like(gradim0)

    a = np.max(gradim0)
    b = np.max(gradim1)

    gradim0 = gradim0.astype('float32')
    gradim1 = gradim1.astype('float32')
    
    gradim0 = (gradim0 * 255) /a
    gradim1 = (gradim1 * 255) /b

    calib_pic_arr = np.dstack((gradim0,gradim1,gradim2))
    calib_pic_arr = calib_pic_arr.astype('uint8')
    
    return calib_pic_arr
    
#??? Not sure what this does
def grad_match(gradA,gradB):
    
    gradC = np.zeros_like(gradA)
    
    calib_pic_arr = np.dstack((gradA,gradB,gradC))
    calib_pic_arr = calib_pic_arr.astype('uint8')
    return calib_pic_arr

#Function to quickly chekc the alignment of the images
def check_alignment(image_310, image_330, 
                    shift = np.array([[1,0,0],[0,1,0]],dtype='float32') ):
    alig_image_330 = alignment(image_330,shift)
    plt.figure()
    plt.imshow(grad_im(image_310,alig_image_330))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return 
