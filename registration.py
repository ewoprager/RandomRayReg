import torch
from typing import Union, TypeAlias
import matplotlib.pyplot as plt

import debug

class Registration:
    """
    A 1D/2D or 2D/3D radiographic image registration task.
    """
    def __init__(self,
                 ray_type: TypeAlias,
                 image_type: TypeAlias,
                 volume,
                 *,
                 source_position: torch.Tensor):
        self.ray_type = ray_type
        self.image_type = image_type
        self.volume = volume
        self.source_position = source_position
        self.image: Union[image_type, None] = None

    def set_image(self, image):
        assert (isinstance(image, self.image_type)), "Invalid type of image given"
        self.image = image

    def set_image_from_random_drr(self,
                                  *,
                                  image_size: torch.Tensor,
                                  drr_alpha: float):
        true_theta = self.ray_type.Transformation()
        true_theta.randomise()
        debug.tic("Generating new DRR at random theta = {}".format(true_theta))
        self.set_image(self.image_type(self.generate_drr(true_theta, image_size=image_size, alpha=drr_alpha), generate_mip_levels=True))
        debug.toc()
        return true_theta

    def set_image_from_drr(self,
                           theta,
                           *,
                           image_size: torch.Tensor,
                           drr_alpha: float):
        assert (isinstance(theta, self.ray_type.Transformation)), "Invalid type of theta given"
        debug.tic("Generating new DRR at theta = {}".format(theta))
        self.set_image(self.image_type(self.generate_drr(theta, image_size=image_size, alpha=drr_alpha), generate_mip_levels=True))
        debug.toc()

    def save_image(self, cache_directory: str):
        torch.save(self.image, cache_directory + "/image.pt")

    def load_image(self, cache_directory: str):
        self.image = torch.load(cache_directory + "/image.pt")
        assert (isinstance(self.image, self.image_type)), "Invalid type of image loaded"

    def generate_drr(self,
                     theta,
                     *,
                     alpha: float,
                     image_size: torch.Tensor,
                     mip_level: int=0) -> torch.Tensor:
        """
        :param theta: Transformation of the DRR, of type `self.ray_type.Transformation`
        :return: A DRR through the stored CT volume at the given transformation, `theta`.
        """
        untransformed_rays = self.ray_type.generate_true_untransformed(image_size, self.source_position, device=self.volume.data[mip_level].device)
        drr_rays = self.ray_type.transform(untransformed_rays, theta.inverse())
        drr_data = self.volume.integrate(drr_rays, alpha=alpha, mip_level=mip_level)

        ## for debugging DRR generation; the produced plot should match the DRR image
        #positions, _ = self.ray_type.xy_plane_intersections(untransformed_rays)
        #plt.scatter(positions[:, 0], positions[:, 1], c=drr_data, s=0.3)
        #plt.axis('square')
        #plt.show()
        ##

        return drr_data.reshape(tuple(image_size)).t()
