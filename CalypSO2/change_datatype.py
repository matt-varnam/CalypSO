# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:20:56 2017

This is a simple module simply for converting between 16bit and 8bit images, 
  mostly because some cv2 commands only accept float32 or uint8 as inputs

@author:  Matthew Varnam - mvarnam@hotmail.co.uk - The University of Manchester
"""
#Convert a 16-bit image into an 8-bit image
def uint16_uint8(image):
    if str(image.dtype) == 'uint8':
        print("WARNING - Image is already type uint8")
        return image
    converted_image = image.astype('float32')
    print (converted_image)
    converted_image = converted_image*(256/65536)
    converted_image = converted_image.astype('uint8')
    return converted_image
    
#Convert an 8-bit image into a 16-bit image
def uint8_uint16(image):
    if str(image.dtype) == 'uint16':
        print("WARNING - Image is already type uint16")
        return image  
    converted_image = image.astype('float32')
    converted_image = int(converted_image*(65536/256))
    converted_image = converted_image.astype('uint16')
    return converted_image