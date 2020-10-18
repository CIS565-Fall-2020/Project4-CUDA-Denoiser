CUDA Pathtracing Denoiser
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Jilin Liu
  * [LinkedIn](https://www.linkedin.com/in/jilin-liu-61b273192/), [twitter](https://twitter.com/Jilin18043110).
* Tested on: Windows 10, i7-8750H @ 2.20GHz, 16GB, GTX 1050Ti 4096MB (personal)

## Features and Results

This is a GPU-based Denoiser for Monte Carlo path-traced images implemented in C++ and CUDA. 
Denoisers can help produce a smoother appearance in a pathtraced image with fewer samples-per-pixel/iterations.

![](./img/compareDenoise.JPG)

The technique is based on "Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering," by Dammertz, Sewtz, Hanika, and Lensch. You can find the paper here: https://jo.dreggn.org/home/2010_atrous.pdf. The raytracing part is based on [this repo](https://github.com/Songsong97/Project3-CUDA-Path-Tracer).

Features:
1. Basic denoiser for Monte-Carlo ray-traced images.
2. Preserve sharp edges for rendered image by using G-Buffer.

## Performance Analysis

