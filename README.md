CUDA Denoiser For CUDA Path Tracer
==================================


**Denoising - Before and After**

<img src="renders/raw2.png" width="300"> <img src="renders/denoise2.png" width="300">

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

NAME: CHETAN PARTIBAN 

GPU: GTX 970m (Compute Capability 5.2) 

Tested on Windows 10, i7-6700HQ @ 2.60 GHz 16Gb, GTX 970m 6Gb (Personal Laptop) 

## Documentation

In this project I impelemented the paper "Edge-Avoiding À-Trous Wavelet Transform for fast Global Illumination Filtering" to denoise path-traced images with a low number of samples per pixel. This process starts by conducting a few iterations of sending out rays to trace in the environment to create an initial "raw" ray traced image as well as by caching a few of the important first-bounce characteristics -- namely the normal vector and intersection point of the first bounce. We can see some examples of those three input buffers to the algorithm visualized below. From left to right we see the raw ray traced image, the position vectors, and the normal vectors. 

<img src="renders/raw10.png" width="200"> <img src="renders/positions.PNG" width="200"> <img src="renders/normals.PNG" width="200">

The "À-Trous Wavelet Transform" component of the algorithm comes from using the À-Trous approximation to a convolving a gaussian kernel with the image. This involves doing a sort of iterative approximation. We can compare it to the opencv implementation of gaussain blur below (on the left is opencv gaussian blur and on the left is our À-Trous Wavelet Transform Blur. We can see the visual effect of the blurring looks pretty similar across the images, indicating that the approximation seems to be accurate enough for our use case.

<img src="renders/opencv.PNG" width="300"> <img src="renders/unweighted.png" width="300">

In order to make the blurring look more acceptable we introduce an edge aware version of the blurring. In this version, as we convolve our À-Trous filter, we add additional weighting to down-weight samples that have very different color/normals/position when compared to the center pixel that the filter is going to write back into. When we add these rules we get more clear edges across objects whilst still blurring within the object. We can see this result below (left is raw image, right is the denoised version). These results were collected at 10 samples per pixel.

<img src="renders/raw10.png" width="300"> <img src="renders/denoise10.png" width="300">

One thing that we can imediately see is that this method performs very well for diffuse surfaces but tends to fail in surfaces that are specular/refractive as any of the effects get blurred together because we are only looking at the first bounce cache of the position/normal vectors. However, still, we can see that we do a good job at removing "holes" from the ray tracing noise whilst still maintaining clear definition between the different objects. 
