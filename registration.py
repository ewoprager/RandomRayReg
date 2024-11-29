import torch

import ray
import data


class Registration:
    def __init__(self,
                 volume: data.Volume,
                 image_size: int,
                 source_position: torch.Tensor=torch.tensor([5., 0.])):
        self.volume = volume
        self.image_size = image_size
        self.source_position = source_position

        self.true_theta = torch.pi * (-1. + 2. * torch.rand(1))
        self.image = self.generate_drr(self.true_theta)

    def generate_drr(self, theta: torch.Tensor) -> data.Image:
        drr_rays = ray.transform(ray.generate_true_untransformed(self.image_size, self.source_position), theta)
        drr_data = self.volume.integrate(drr_rays)
        return data.Image(drr_data)