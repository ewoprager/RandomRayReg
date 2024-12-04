from typing import Union, Tuple
import numpy as np
import torch

import tools
from one_d_two_d.ray import Ray

class Volume:
    """
    A 2D image that rays can be integrated along through
    """
    def __init__(self, data: torch.Tensor):
        self.data = data
        self.size = data.size()

    def samples(self, positions: torch.Tensor) -> torch.Tensor:
        """
        :param positions: a vector of 2D vectors, positions between (-1,-1) and (1,1)
        :return: vector of bi-linearly interpolated samples from the stored image at the given positions
        """
        return torch.nn.functional.grid_sample(self.data[None, None, :, :], positions[None, None, :, :], align_corners=False)[0, 0, 0]

    def integrate(self, rays: torch.Tensor, n: int=200, alpha: float=0.5) -> torch.Tensor:
        """
        :param rays: tensor of rays to integrate along
        :param n: The number of points to sample along each ray
        :param alpha: X-ray attenuation factor
        :return: A tensor of approximations of the X-ray intensities attenuated along the given rays through the CT
                 volume. This is calculated as `1 - exp(-alpha * sum)` where `sum` is the approximate average value
                 along rays in the CT volume.
        """
        perpendiculars = tools.cross_vectors2d(rays[:, 2:4])
        offsets = (rays[:, 0:2] * perpendiculars).sum(dim=1)[:, None]
        ps = -np.sqrt(2.) * rays[:, 2:4] + offsets * perpendiculars
        deltas = (2. * np.sqrt(2.) / float(n)) * rays[:, 2:4]
        ret = torch.zeros(rays.size()[0])
        for i in range(n):
            ret += self.samples(ps)
            ps += deltas
        return 1. - torch.exp(-ret / (alpha * float(n)))

    def display(self, axes):
        xs, ys = np.meshgrid(np.linspace(-1., 1., self.size[0], endpoint=True), np.linspace(-1., 1., self.size[1], endpoint=True))
        axes.pcolormesh(xs, ys, self.data, cmap='gray')


class Image:
    """
    A 1D image that can be sampled using rays
    """
    def __init__(self,
                 data: torch.Tensor):
        self.data = data
        self.size = data.size()

    def samples(self, rays: torch.Tensor, blur_sigma: Union[torch.Tensor, None]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolates samples from the stored image where the given rays intersect the y-axis.
        :param rays: tensor of rays
        :param blur_sigma: (optional) sigma with which to apply a Gaussian blur to the image before sampling
        :return: tensor of samples for each ray, tensor of weight modifications for each ray
        """
        data = self.data if blur_sigma is None else tools.gaussian_blur1d(self.data, blur_sigma.item())
        positions = Ray.y_axis_intersections(rays)
        positions_transformed = .5 * (positions + 1.) * (torch.tensor(self.size, dtype=torch.float32) - 1.)
        i0s = torch.floor(positions_transformed.clone().detach()).type(torch.int64)
        fs = positions_transformed - i0s.type(torch.float32)
        with_zero = torch.cat((torch.zeros(1), data))
        i0s = i0s + 1
        i1s = i0s + 1
        n = with_zero.size()[0]
        i0s_out = torch.logical_or(i0s < 1, i0s >= n)
        i1s_out = torch.logical_or(i1s < 1, i1s >= n)
        i0s[i0s_out] = 0
        i1s[i1s_out] = 0

        # determining image-edge weight modifications
        weights = (1. - fs) * torch.logical_not(i0s_out).type(torch.float32) + fs * torch.logical_not(i1s_out).type(torch.float32)

        # sampling
        ret = (1. - fs) * with_zero.gather(0, i0s) + fs * with_zero.gather(0, i1s)

        return ret, weights

    def display(self, axes):
        axes.pcolormesh(self.data[None, :], cmap='gray')