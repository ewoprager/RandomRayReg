import torch

import one_d_two_d.ray as ray
import one_d_two_d.data as data
from one_d_two_d.ray import Transformation

class Registration:
    """
    A 1D/2D radiographic image registration task.
    """
    def __init__(self,
                 volume: data.Volume,
                 image_size: int,
                 source_position: torch.Tensor):
        self.volume = volume
        self.image_size = image_size
        self.source_position = source_position

        self.true_theta: Transformation = Transformation()
        self.true_theta.randomise()
        self.image: data.Image = self.generate_drr(self.true_theta)

    def generate_drr(self, theta: Transformation) -> data.Image:
        """
        :param theta: Transformation of the DRR
        :return: A DRR through the stored CT volume at the given transformation, `theta`.
        """
        drr_rays = ray.transform(ray.generate_true_untransformed(self.image_size, self.source_position), theta)
        drr_data = self.volume.integrate(drr_rays)
        return data.Image(drr_data)