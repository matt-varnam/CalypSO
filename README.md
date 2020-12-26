# CalypSO

This program uses satellite DEMs from the Shuttle Radar Topography Mission (SRTM) and knowledge of a camera's location and properties to predict the skyline visible in the camera image. The predicted skyline then compares to the real skyline as captured in the camera imagery to iterate the camera viewing direction parameters, though the initial guess must be within 1 degree got the automatic iteration to function.

When a good match to the real skyline is achieved, the program then outputs the distance to the ground for each pixel in the image.

It is intended for use with SO<sub>2</sub> cameras to automatically generate the distance to each point on a volcanic ediface. This can then be used to perform a light dilution correction for the SO<sub>2</sub> emission rate measured.

# How it works

The skyline is identified using the brightness contrast in the image between the sky and the ground. 

The creation of the virtual skyline identifies the highest pixel in each column that can see the ground.

The iteration uses SciPy's bounded COBYLA minimisation to avoid the iteration finding an alternate minimum (though this still happens a bit)

All other code is geometry calculations and transformations.

# Contained programs

plot_geotiff.py aligns the virtual skyline with a real skyline, then outputs the distance to the ground for each pixel

plume_distance.py can be used to calculate the distance to an SO<sub>2</sub> at each pixel plume blowing from a defined latitude and longitude, providing simultaneous traverses are used (this section may require reworking for volcanoes other than Masaya)
