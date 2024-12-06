import torch
from typing import Union, TypeAlias
import matplotlib.pyplot as plt

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
        self.set_image(self.generate_drr(true_theta, image_size=image_size, alpha=drr_alpha))
        return true_theta

    def set_image_from_drr(self,
                           theta,
                           *,
                           image_size: torch.Tensor,
                           drr_alpha: float):
        assert (isinstance(theta, self.ray_type.Transformation)), "Invalid type of theta given"
        self.set_image(self.generate_drr(theta, image_size=image_size, alpha=drr_alpha))

    def save_image(self, cache_directory: str):
        torch.save(self.image, cache_directory + "/image.pt")

    def load_image(self, cache_directory: str):
        self.image = torch.load(cache_directory + "/image.pt")
        assert (isinstance(self.image, self.image_type)), "Invalid type of image loaded"

    def generate_drr(self,
                     theta,
                     *,
                     alpha: float,
                     image_size: torch.Tensor):
        """
        :param theta: Transformation of the DRR, of type `self.ray.Transformation`
        :return: A DRR through the stored CT volume at the given transformation, `theta`.
        """
        untransformed_rays = self.ray_type.generate_true_untransformed(image_size, self.source_position, device=self.volume.data.device)
        drr_rays = self.ray_type.transform(untransformed_rays, theta.inverse())
        drr_data = self.volume.integrate(drr_rays, alpha=alpha)
        drr_data[drr_data.isnan()] = 0.

        ## for debugging DRR generation; the produced plot should match the DRR image
        #positions, _ = self.ray_type.xy_plane_intersections(untransformed_rays)
        #plt.scatter(positions[:, 0], positions[:, 1], c=drr_data, s=0.3)
        #plt.axis('square')
        #plt.show()
        ##

        return self.image_type(drr_data.reshape(tuple(image_size)).t())
