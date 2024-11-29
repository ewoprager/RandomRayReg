import torch

import ray
import tools
from registration import Registration


class RandomRays:
    """
    A number of rays generated at random transformations and integrated along in a CT volume to be used for registration
    """
    def __init__(self, registration: Registration, ray_count: int=1000):
        self.registration = registration
        self.ray_count = ray_count

        # rays are defined by two 2D vectors: an origin and direction in CT space
        self.data = ray.generate_random(self.ray_count)
        self.intensities = self.registration.volume.integrate(self.data)

    def plot_with_intensity_shading(self, axes):
        """
        Plot the stored rays on the given axes, colouring them according to their integral through the CT volume
        :param axes:
        """
        min_intensity = self.intensities.min()
        intensity_range = self.intensities.max() - min_intensity
        shades = (self.intensities - min_intensity) / intensity_range
        ray.plot(axes, self.data, shades)

    def plot_with_sampled_shading(self, axes, theta=None):
        """
        Plot the stored rays on the given axes, colouring them according to their fixed image samples at their
        intersections with the y-axis
        :param axes:
        :param theta: (optional) transformation to apply to the rays before plotting
        """
        transformed_rays = self.data if theta is None else ray.transform(self.data, theta)
        ray.plot(axes, transformed_rays, self.registration.image.samples(transformed_rays))

    def evaluate(self, theta: torch.Tensor, alpha: torch.Tensor=torch.tensor([0.14]), blur_constant: torch.Tensor=torch.Tensor([1.])) -> (torch.Tensor, torch.Tensor):
        """
        Find the approximation of the ZNCC between the fixed image and a DRR at the given transformation, theta, using
        the stored pre-integrated rays.
        :param theta: Transformation to apply to the rays before evaluation
        :param alpha: Distance drop-off coefficient used for calculating ray weights
        :param blur_constant: Coefficient for transforming alpha into the sigma used to blur the fixed image
        :return: (the ZNCC of the calculated intensity pairs, the sum of all pair weights)
        """
        transformed_rays = ray.transform(self.data, theta)
        # scoring rays on distance from source
        scores = ray.scores(transformed_rays, self.registration.source_position, alpha=alpha)
        # sampling from blurred image for every ray
        normalised_alpha = alpha / torch.norm(self.registration.source_position)
        blur_sigma = blur_constant * normalised_alpha
        samples = self.registration.image.samples(transformed_rays, blur_sigma=blur_sigma)
        # calculating similarity between samples and intensities, weighted by scores
        negative_zncc = -tools.weighted_zero_normalised_cross_correlation(samples, self.intensities, scores)
        sum_n = scores.sum()
        return negative_zncc, sum_n

    def evaluate_with_grad(self, theta: torch.Tensor, alpha: torch.Tensor=torch.tensor([0.14]), blur_constant: torch.Tensor=torch.Tensor([1.])) -> (torch.Tensor, torch.Tensor):
        """
        Find the approximation of the ZNCC between the fixed image and a DRR at the given transformation, theta, using
        the stored pre-integrated rays, with autograd enabled for use in directed search.
        :param theta: Transformation to apply to the rays before evaluation
        :param alpha: Distance drop-off coefficient used for calculating ray weights
        :param blur_constant: Coefficient for transforming alpha into the sigma used to blur the fixed image
        :return: (the ZNCC of the calculated intensity pairs, the sum of all pair weights)
        """
        theta.grad = torch.zeros_like(theta)
        theta.requires_grad_(True)
        negative_zncc, sum_n = self.evaluate(theta, alpha=alpha, blur_constant=blur_constant)
        negative_zncc.backward(torch.ones_like(negative_zncc))
        theta.requires_grad_(False)
        return negative_zncc, sum_n

