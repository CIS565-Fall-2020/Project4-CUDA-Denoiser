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
1. From the results above, we see that with denoising, we obtain an acceptably smooth results in 20 iterations, while without denoising, we need about 850 iterations.

2. From the table below, we see the run time per iteration nearly doubled with denoising.

|                        | with denoising | without denoising |
| ---                    | ---            | ---               |
| Run Time Per Iteration | 120.964        |  58.2717          |

3. Denoising is more effective on object with diffuse material then reflective or refractive materials. This is because diffuse surface reflect light randomly, causing an uneven distribution of pixels getting mapped to image in those areas. Refractive or reflective materials, on the other hand, reflect light more deterministically. Therefore, the diffuse surfaces would benefit more from gaussian blurring. 

4. Visual and run time comparison of different filter sizes.

We observe that visual results improve greatly from filtersize=10 to 20 and from 20 to 40. At smaller filtersize, the image is less smooth and we can see blocks of color. The image quality is stable after filtersize=40. 

We observe an increase in run time per iteration as we increase filtersize. However, the increase becomes smaller when filtersize is large. 

| Filter Size            | 10             | 20                | 40      | 80      |
| ---                    | ---            | ---               | ---     | ---     |
| Run Time Per Iteration |   91.114       |    104.181        | 120.628 | 122.903 |
| Visual Result          |    <img src="https://github.com/WeiyuDu/Project4-CUDA-Denoiser/blob/denoiser/img/filter_10.png" width=150/>     |      <img src="https://github.com/WeiyuDu/Project4-CUDA-Denoiser/blob/denoiser/img/filter_20.png" width=150/>      |  <img src="https://github.com/WeiyuDu/Project4-CUDA-Denoiser/blob/denoiser/img/filter_40.png" width=150/>  | <img src="https://github.com/WeiyuDu/Project4-CUDA-Denoiser/blob/denoiser/img/filter_80.png" width=150/>|

5. Visual comparison of different scenes. Left: denoised image; Right: original image. (Iteration = 20, same parameter is used as the above cornell ceiling light scene.) We observe that we do not achieve as good of a denoised result as the cornel ceiling light scene. This may be because cornell ceiling light has a large light source. When the number of iterations is low, more pixels got filled in the ceiling light scene than in other two scenes. The original images from the two scenes are at a worse quality, therefore the denoised images are at a worse quality.

<nobr><img src="https://github.com/WeiyuDu/Project4-CUDA-Denoiser/blob/denoiser/img/cornell_denoised.png" width=300/></nobr>
<nobr><img src="https://github.com/WeiyuDu/Project4-CUDA-Denoiser/blob/denoiser/img/cornell_orig.png" width=300/></nobr>

<nobr><img src="https://github.com/WeiyuDu/Project4-CUDA-Denoiser/blob/denoiser/img/refractive_denoised.png" width=300/></nobr>
<nobr><img src="https://github.com/WeiyuDu/Project4-CUDA-Denoiser/blob/denoiser/img/refractive_orig.png" width=300/></nobr>

6. Run time comparison of denoising at different image resolutions. We observe as image resolution increases, the run time per iteration also increases. The run time for rendering denoised image is almost proportional to the number of pixels.

| Image Resolution       | 200 x 200      | 400 x 400         | 600 x 600      | 800 x 800      |
| ---                    | ---            | ---               | ---            | ---            |
| Run Time Per Iteration |   12.8942      |    36.0212        | 70.0918        | 120.405        |

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
