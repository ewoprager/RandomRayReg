# Radiographic 2D/3D image registration with random rays

Experimental implementation of 2D/3D image registration method, with a moving 3D image and a fixed 2D image.

The analogous 1D/2D image registration task is also implemented.

See [RandomRays.pdf](RandomRays.pdf) for a detailed motivation, derivation and explanation.

# Modules

### `tools`

Some simple mathematical operations on images and vectors.


### `registration`

A class holding information about a registration problem.


### `random_rays`

A class implementing functionality for solving a registration problem using random rays.


## `two_d_three_d`

### `two_d_three_d.ray`

A ray is stored as a row of a 6-column tensor. The first 3 values are a position in CT space through which the ray
passes, and the final 3 values are a unit-length direction vector of the ray.

This module contains some basic functions for manipulating rays stored as rows of `torch.Tensor`s.


### `two_d_three_d.data`

Classes for holding 2D and 3D images, with corresponding sampling with rays.


## `one_d_two_d`

### `one_d_two_d.ray`

A ray is stored as a row of a 4-column tensor. The first 2 values are a position in CT space through which the ray
passes, and the final 2 values are a unit-length direction vector of the ray.

This module contains some basic functions for manipulating rays stored as rows of `torch.Tensor`s.


### `one_d_two_d.data`

Classes for holding 1D and 2D images, with corresponding sampling with rays.


# Scripts

## `one_d_two_d.main`

Sets up a registration problem and implements workflows for
- visualising the optimisation landscape, and
- registering using an optimiser.

### Example output:

For a randomly generated CT volume:

![ct.png](one_d_two_d/plots/ct.png)

And a randomly chosen orientation: `-0.188 radians`, giving the following DRR through the volume:

![drr_true.png](one_d_two_d/plots/drr_true.png)

The optimisation landscape was evaluated using the random rays technique with hyperparameters as follows:
- `ray_density = 1000.` for a ray count of `5318`
- `blur_constant = 4.`

with different values of alpha (differentiated by colour), and with/without clipping rays to within the X-ray image
(dashed=clipped):

![landscape.png](one_d_two_d/plots/landscape.png)

Running ASGD on this from a random starting position converged on an orientation of `0.164 radians`, which managed to be
in the same local minimum as the global minimum. A DRR taken at this orientation is as follows:

![drr_final.png](one_d_two_d/plots/drr_final.png)

The WZNCC and orientation developed as follows over the course of the optimisation:

![optimisation.png](one_d_two_d/plots/optimisation.png)


# Setup

Developed with `Python 3.12.4`.

Install requirements with
```bash
pip install -r requirements.txt
```