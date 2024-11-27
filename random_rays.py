import torch

import ray
import tools
from registration import Registration


class RandomRays:
    def __init__(self, registration: Registration, ray_count: int=1000):
        self.registration = registration
        self.ray_count = ray_count

        # rays are defined by two 2D vectors: an origin and direction in CT space
        self.data = ray.generate_random(self.ray_count)
        self.intensities = self.registration.volume.integrate(self.data)

    def plot_with_intensity_shading(self, axes):
        min_intensity = self.intensities.min()
        intensity_range = self.intensities.max() - min_intensity
        shades = (self.intensities - min_intensity) / intensity_range
        ray.plot(axes, self.data, shades)

    def plot_with_sampled_shading(self, axes, theta=None):
        transformed_rays = self.data if theta is None else ray.transform(self.data, theta)
        ray.plot(axes, transformed_rays, self.registration.image.samples(transformed_rays))

    def evaluate(self, theta: torch.Tensor, alpha: torch.Tensor=torch.tensor([50.])) -> (torch.Tensor, torch.Tensor):
        transformed_rays = ray.transform(self.data, theta)
        scores = ray.scores(transformed_rays, self.registration.source_position, alpha=alpha)
        samples = self.registration.image.samples(transformed_rays)
        negative_zncc = -tools.zero_normalised_cross_correlation(samples, self.intensities, scores)
        sum_n = scores.sum()
        return negative_zncc, sum_n

    def evaluate_with_grad(self, theta: torch.Tensor, alpha: torch.Tensor=torch.tensor([50.])) -> (torch.Tensor, torch.Tensor):
        theta.grad = torch.zeros_like(theta)
        theta.requires_grad_(True)
        negative_zncc, sum_n = self.evaluate(theta, alpha=alpha)
        negative_zncc.backward(torch.ones_like(negative_zncc))
        theta.requires_grad_(False)
        return negative_zncc, sum_n

