CUDA Denoiser
================
**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**
* Haorong Yang
* [LinkedIn](https://www.linkedin.com/in/haorong-henry-yang/)
* Tested on: Windows 10 Home, i7-10750H @ 2.60GHz 16GB, GTX 2070 Super Max-Q (Personal)

<img src="img/smoothed2.png" width="650">  



### Overview:
This is an implementation of A Trous filter denoising based on the paper: [Edge-Avoiding À-Trous Wavelet Transform for fast Global
Illumination Filtering](https://jo.dreggn.org/home/2010_atrous.pdf). The reference images are performed on a basic Monte-Carlo Path Tracer.
  

### Denoise Results
Unsmoothed           |       Smoothed           
:-------------------------:|:-------------------------:
<img src="img/unsmoothed.png" width="500">| <img src="img/smoothed.png" width="500"> |


### GBuffer visualizations
positions         |   normals
:-------------------------:|:-------------------------:
<img src="img/pos.png" width="500">| <img src="img/normal.png" width="500"> |


### Performance
<img src="img/performance.PNG" width="500">


Here is a comparison of the rendering time with and without denoising. We can see that denoising always will take up extra processing time, and increases linearly with filter size.

