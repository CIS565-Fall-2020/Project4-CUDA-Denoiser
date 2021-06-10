CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Tushar Purang
* Tested on Windows 10, i7-7700HQ CPU @ 2.80 GHz 16Gb, GTX 1060 6Gb (Personal Laptop) 

## Documentation

In this project, I implemented the paper "Edge-avoiding À-Trous wavelet transform for fast global illumination filtering" for denoising images with low samples per pixel. This algorithm involves using the À-Trous approximation to convolve a gaussian kernel with the sampled image. 

<img src="\finalImages\cornell.2021-06-09_23-59-49z.10samp.png" style="zoom:50%;" />   Raw Path-traced image



<img src="\finalImages\cornell.2021-06-10_00-00-43z.11samp.png" style="zoom:50%;" /> Sample smoothed image



In order to make the blurring look more acceptable we introduce an edge aware version of the blurring. In this version, as we convolve our À-Trous filter, we add additional weighting to down-weight samples that have very different color/normals/position when compared to the center pixel that the filter is going to write back into. When we add these rules we get more clear edges across objects whilst still blurring within the object.

<img src="\finalImages\cornell.2021-06-10_00-07-24z.11samp.png" style="zoom:50%;" />   Unweighted smoothed image

<img src="\finalImages\cornell.2021-06-10_00-03-08z.11samp.png" style="zoom:50%;" />   Weighted smoothed image

It can be observed that this method performs well for diffuse surfaces but fails for specular/refractive surfaces. The produced effects get blurred together because we are only looking at the first bounce cache of the position/normal vectors. 