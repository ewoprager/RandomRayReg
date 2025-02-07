import torch
from typing import Union
import matplotlib.pyplot as plt

import tools
from registration import Registration


class RandomRays:
    """
    A number of rays generated at random transformations and integrated along in a CT volume to be used for registration
    """
    __init_key = object()

    def __init__(self, init_key, registration: Registration, data: torch.Tensor, intensities: torch.Tensor):
        """
        Private constructor
        :param init_key:
        :param registration: Registration task object
        :param data: tensor of rays
        :param intensities: tensor of pre-calculated radiological paths for the rays
        """
        assert (init_key is self.__class__.__init_key), "Constructor is private"
        self.registration = registration
        self.ray_type = self.registration.ray_type
        self.data = data
        self.ray_count = self.data.size()[0]
        self.intensities = intensities

    @classmethod
    def new(cls, registration: Registration, *, integrate_alpha: float, ray_count: int = 1000):
        data = registration.ray_type.generate_random(ray_count)
        intensities = registration.volume.integrate(data, alpha=integrate_alpha)
        return cls(cls.__init_key, registration, data, intensities)

    @classmethod
    def load(cls, registration: Registration, cache_directory: str):
        data = torch.load(cache_directory + "/rays.pt")
        intensities = torch.load(cache_directory + "/intensities.pt")
        return cls(cls.__init_key, registration, data, intensities)

    def save(self, cache_directory: str):
        torch.save(self.data, cache_directory + "/rays.pt")
        torch.save(self.intensities, cache_directory + "/intensities.pt")

    def plot_with_intensity_shading(self, axes):
        """
        Plot the stored rays on the given axes, colouring them according to their integral through the CT volume
        :param axes:
        """
        min_intensity = self.intensities.min()
        intensity_range = self.intensities.max() - min_intensity
        shades = (self.intensities - min_intensity) / intensity_range
        self.ray_type.plot(axes, self.data, shades)

    def plot_with_sampled_shading(self, axes, theta=None):
        """
        Plot the stored rays on the given axes, colouring them according to their fixed image samples at their
        intersections with the y-axis
        :param axes:
        :param theta: (optional) transformation to apply to the rays before plotting
        """
        transformed_rays = self.data if theta is None else self.ray_type.transform(self.data, theta)
        self.ray_type.plot(axes, transformed_rays, self.registration.image.samples(transformed_rays)[0])

    def evaluate(self, theta, alpha: torch.Tensor = torch.tensor([0.14]),
                 blur_constant: torch.Tensor = torch.Tensor([1.]), clip: bool = True,
                 ray_count: Union[int, None] = None, debug_plots: bool = False) -> (torch.Tensor, torch.Tensor):
        """
        Find the approximation of the ZNCC between the fixed image and a DRR at the given transformation, theta, using
        the stored pre-integrated rays.
        :param theta: Transformation to apply to the rays before evaluation
        :param alpha: Distance drop-off coefficient used for calculating ray weights
        :param blur_constant: Coefficient for transforming alpha into the sigma used to blur the fixed image
        :param clip: whether to modify ray weights that don't intersect the X-ray detector array
        :param ray_count: (optional) number of rays to use in calculation, if not all of them
        :return: (the ZNCC of the calculated intensity pairs, the sum of all pair weights)
        """
        transformed_rays = self.ray_type.transform(self.data if ray_count is None else self.data[0:ray_count], theta)
        # sampling from blurred image for every ray
        normalised_alpha = alpha / torch.norm(self.registration.source_position)
        blur_sigma = blur_constant * normalised_alpha
        samples, weight_modifications, positions = self.registration.image.samples(transformed_rays,
            blur_sigma=blur_sigma)
        # scoring rays on distance from source
        scores = self.ray_type.scores(transformed_rays, self.registration.source_position, alpha=alpha)
        if clip:
            scores = scores * weight_modifications

        ## plotting pseudo-image
        if debug_plots:
            plot_indices = torch.logical_and(positions.abs().max(dim=-1)[0] < 1., scores > 0.3)
            plot_positions = positions[plot_indices]
            plot_samples = samples[plot_indices]
            plot_intensities = (self.intensities if ray_count is None else self.intensities[0:ray_count])[plot_indices]
            plt.scatter(plot_positions[:, 0], plot_positions[:, 1], c=plot_intensities, s=0.3)
            plt.axis('square')
            plt.show()
            plt.scatter(plot_positions[:, 0], plot_positions[:, 1], c=plot_samples, s=0.3)
            plt.axis('square')
            plt.show()
        ##

        # calculating similarity between samples and intensities, weighted by scores
        negative_zncc = -tools.weighted_zero_normalised_cross_correlation(samples,
            self.intensities if ray_count is None else self.intensities[0:ray_count], scores)
        sum_n = scores.sum()
        return negative_zncc, sum_n

    def evaluate_with_grad(self, theta, alpha: torch.Tensor = torch.tensor([0.14]),
                           blur_constant: torch.Tensor = torch.Tensor([1.]), clip: bool = True,
                           ray_count: Union[int, None] = None) -> (torch.Tensor, torch.Tensor):
        """
        Find the approximation of the ZNCC between the fixed image and a DRR at the given transformation, theta, using
        the stored pre-integrated rays, with autograd enabled for use in directed search.
        :param theta: Transformation to apply to the rays before evaluation
        :param alpha: Distance drop-off coefficient used for calculating ray weights
        :param blur_constant: Coefficient for transforming alpha into the sigma used to blur the fixed image
        :param clip: whether to modify ray weights that don't intersect the X-ray detector array
        :param ray_count: (optional) number of rays to use in calculation, if not all of them
        :return: (the ZNCC of the calculated intensity pairs, the sum of all pair weights)
        """
        theta.enable_grad()
        negative_zncc, sum_n = self.evaluate(theta, alpha=alpha, blur_constant=blur_constant, clip=clip,
            ray_count=ray_count)
        negative_zncc.backward(torch.ones_like(negative_zncc))
        theta.disable_grad()
        return negative_zncc, sum_n
