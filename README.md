CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

### Performance analysis

#### Render time 
For the cornell box scene, we measure the average time per iteration, which is quite stable.

#### Denoise time
To analyze the time used to denoise, first we consider the image of 800 * 800 resolution, we change the number of filter levels. We could observe that the time needed to denoise increases for the first levels but remain stable when the stepwidth gets too big, which is expected. Compare the denoise time with the render timer per iteration we get from the previous section. The denoise time is roughly equal to one iteration for the cornell box scene.

#### Denoise performance
To get a roughly smooth scene, we need ~150 iterations for the cornell box scene.


