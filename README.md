CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Weiyu Du
* Tested on: CETS Virtual Lab
### Part 3
### Denoiser
Left: denoised image; Right: original image. (iteration=20)

<nobr><img src="https://github.com/WeiyuDu/Project4-CUDA-Denoiser/blob/denoiser/img/denoised_20.png" width=300/></nobr>
<nobr><img src="https://github.com/WeiyuDu/Project4-CUDA-Denoiser/blob/denoiser/img/original_20.png" width=300/></nobr>

### Performance Analysis
1. From the results above, we see that with denoising, we obtain an acceptably smooth results in 20 iterations, while without denoising, we need about 1000 iterations.

2. From the table below, we see the run time per iteration nearly doubled with denoising.

|                        | with denoising | without denoising |
| ---                    | ---            | ---               |
| Run Time Per Iteration | 120.964        |  58.2717          |

3. Denoising is more effective on object with diffuse material then reflective or refractive materials.

|                        | 20             | 40                | 60      | 80      |
| ---                    | ---            | ---               | ---     | ---     |
| Run Time Per Iteration | 104.181        |  120.628          | 120.434 | 119.834 |
| Visual Result          |    <img src="https://github.com/WeiyuDu/Project4-CUDA-Denoiser/blob/denoiser/img/denoised_20.png" width=120/>     |      <img src="https://github.com/WeiyuDu/Project4-CUDA-Denoiser/blob/denoiser/img/denoised_40.png" width=120/>      |  <img src="https://github.com/WeiyuDu/Project4-CUDA-Denoiser/blob/denoiser/img/denoised_60.png" width=120/>  | <img src="https://github.com/WeiyuDu/Project4-CUDA-Denoiser/blob/denoiser/img/denoised_80.png" width=120/>|

how it impacts runtime at different resolutions


Also compare visual results and performance for varying filter sizes.

Be sure to compare across different scenes as well - for example, between cornell.txt and cornell_ceiling_light.txt. Does one scene produce better denoised results? Why or why not?
### Part 2
### Refraction
Refraction rendering with Frensel effects using Schlick's approximation

<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/refract.png" width=300/>

### Depth of Field
From left to right: focus on foreground, focus on background

<nobr><img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/dof_close.png" width=300/>
<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/dof_far.png" width=300/></nobr>

### Stochastic Sampled Antialiasing

<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/antialiasing.png" width=300/>

### Arbitrary OBJ Mesh Loader

<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/wahoo.png" width=300/>
Performance comparison regarding bounding volume interseciton culling (measured in time per iteration):

| OBJ file | bounding volume intersection culling | naive implementation |
| ---      | ---                                  | ---                  |
| Sphere   | 98.122 | 129.479 |
| Wahoo    | 1068.55 | 1453.84 |
| Stanford Bunny | 11970.6 | 22964.9 |

We observe that such optimization reduces the run time per iteration consistenly across different obj files, specifically, the more vertices an obj file has, we observe more significant improvement using bounding volume intersection culling.

### Stratified Sampling

1) Comparison of stratified sampling (10x10 grid, left) and uniform random sampling (right) at 5000 iterations

<nobr><img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/strat_5000.png" width=300/><img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/ref_5000.png" width=300/></nobr>

2) Comparison of stratified sampling (10x10 grid, left) and uniform random sampling (right) at 100 iterations

<nobr><img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/strat_100iter_10x10.png" width=300/><img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/ref_100iter_10x10.png" width=300/></nobr>

### Motion Blur
1) Defined motion in scene file

<nobr><img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/defined_motion1.png" width=300/><img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/defined_motion2.png" width=300/></nobr>

2) User input camera motion (user drag the camera while rendering)

<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/master/img/real_time_motion.png" width=300/>

### Part 1
### Render Result
<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/mid-project-submission/img/render_res.png" width=300/>

### Analysis
1) Plot of elapsed time per iteration versus max ray depth (timed when sorting_material set to true)
<img src="https://github.com/WeiyuDu/Project3-CUDA-Path-Tracer/blob/mid-project-submission/img/hw3_plot.png"/>

- We expected that sorting the rays/path segments by material should improve the performance, because this will make the threads more likely to finish at around the same time, reducing waiting time for threads in the same warp. However, in reality we found that rendering without sorting is actually significantly faster. This may because that there isn't a variety of different materials in the scene. Since we're sorting the entire set of rays, this operation takes much more time than it saves.
- From the plot above we see that increasing max ray depth results in longer run time per iteration. Rendering using first bounce cache is consistently faster than rendering without cache, though not by a large margin. This is expected as we save time by avoiding the initial intersection computation.
