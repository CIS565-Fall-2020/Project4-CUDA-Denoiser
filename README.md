CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Ling Xie
  * [LinkedIn](https://www.linkedin.com/in/ling-xie-94b939182/), 
  * [personal website](https://jack12xl.netlify.app).
* Tested on: 
  * Windows 10, Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz 2.20GHz ( two processors) 
  * 64.0 GB memory
  * NVIDIA TITAN XP GP102

Thanks to [FLARE LAB](http://faculty.sist.shanghaitech.edu.cn/faculty/liuxp/flare/index.html) for this ferocious monster.

##### Cmake change

Add 

1. [PerformanceTimer.h](https://github.com/Jack12xl/Project2-Stream-Compaction/blob/master/src/csvfile.hpp) : Measure performance by system time clock. 
2. [cfg.h](https://github.com/Jack12xl/Project2-Stream-Compaction/blob/master/stream_compaction/radixSort.h),  as a configure

### Intro

This repo contains  a CUDA-based path-tracer Denoiser based on [paper](https://jo.dreggn.org/home/2010_atrous.pdf)(2010) : 

**Edge-Avoiding À-Trous Wavelet Transform for fast Global Illumination Filtering.**

Basically, Denoiser serves to produce a smoother results for path-traced image even under fewer samples. Here we manage to implement the guided filtering which not only smooth the noise but also keep the edge intact.

Different from [Guided filtering](http://kaiminghe.com/eccv10/) (ECCV2010) which only takes semantic mask or itself as guidance(only one).  this paper takes `normal, color and position` as guidance to filter the image. 

!img

### Performance Analysis

##### **Denoising effect:**

From the image showed before, we could see that the 20 samples with denoising applied could achieve comparable results with 5000 samples.

##### Runtime:

Here shows the runtime for each scene

| milliseconds | 5 steps Denoise | No denoise |
| ------------ | --------------- | ---------- |
| Cornell Box  |                 | 330.6      |
| Fresnel      | 1050.5          | 954.842    |

Fresnel scenes:



##### WIth different material types:



##### Different scenes

### GUI modification

Aiming at better debug view, we modify the UI to support switch the frame to visualize between rendered image, normal, position and color.

#### Save by frame texture

Also for better saving, instead of saving by parsing the host image pointer, here we save the image by directly saving the frame texture depicted on screen. The method is convenient both for implementation and practical use. 



### Extra Credit:



### Acknowledge

* [Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering](https://jo.dreggn.org/home/2010_atrous.pdf)
* [Spatiotemporal Variance-Guided Filtering](https://research.nvidia.com/publication/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A)
* [A Survey of Efficient Representations for Independent Unit Vectors](http://jcgt.org/published/0003/02/01/paper.pdf)
* ocornut/imgui - https://github.com/ocornut/imgui
* 