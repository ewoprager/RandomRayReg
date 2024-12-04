import torch
from typing import TypeAlias

class Registration:
    """
    A 1D/2D or 2D/3D radiographic image registration task.
    """
    def __init__(self,
                 ray_type: TypeAlias,
                 image_type: TypeAlias,
                 volume,
                 image_size: torch.Tensor,
                 source_position: torch.Tensor,
                 drr_alpha: float=.5):
        self.ray_type = ray_type
        self.image_type = image_type
        self.volume = volume
        self.image_size = image_size
        self.source_position = source_position

        self.true_theta = self.ray_type.Transformation()
        self.true_theta.randomise()
        self.image = self.generate_drr(self.true_theta, alpha=drr_alpha)

    def generate_drr(self, theta, alpha: float=.5):
        """
        :param theta: Transformation of the DRR, of type `self.ray.Transformation`
        :return: A DRR through the stored CT volume at the given transformation, `theta`.
        """
        drr_rays = self.ray_type.transform(self.ray_type.generate_true_untransformed(self.volume.data.device, self.image_size, self.source_position), theta)
        drr_data = self.volume.integrate(drr_rays, alpha=alpha)
        drr_data[drr_data.isnan()] = 0.
        return self.image_type(drr_data.reshape(tuple(self.image_size)))

    def save(self, cache_directory: str):
        self.image.save(cache_directory + "/image.pt")
