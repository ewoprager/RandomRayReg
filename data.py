from typing import Union
import numpy as np
import torch

import tools

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
        return torch.nn.functional.grid_sample(self.data[None, None, :, :], positions[None, None, :, :], align_corners=False)

    def integrate(self, rays: torch.Tensor, n: int=200, alpha: float=0.5) -> torch.Tensor:
        """
        :param rays: tensor of rays to integrate along
        :param n: The number of points to sample along each ray
        :param alpha: X-ray attenuation factor
        :return: A tensor of approximations of the X-ray intensities attenuated along the given rays through the CT
                 volume. This is calculated as `1 - exp(-alpha * sum)` where `sum` is the approximate average value
                 along rays in the CT volume.
        """
        perps = tools.cross_vectors(rays[:, 2:4])
        offsets = (rays[:, 0:2] * perps).sum(dim=1)[:, None]
        ps = -np.sqrt(2.) * rays[:, 2:4] + offsets * perps
        deltas = (2. * np.sqrt(2.) / float(n)) * rays[:, 2:4]
        ret = torch.zeros(rays.size()[0])
        for i in range(n):
            ret += self.samples(ps)[0, 0, 0]
            ps += deltas
        return 1. - torch.exp(-ret / (alpha * float(n)))

    def display(self, axes):
        X, Y = np.meshgrid(np.linspace(-1., 1., self.size[0], endpoint=True), np.linspace(-1., 1., self.size[1], endpoint=True))
        axes.pcolormesh(X, Y, self.data, cmap='gray')


class Image:
    """
    A 1D image that can be sampled using rays
    """
    def __init__(self,
                 data: torch.Tensor):
        self.data = data
        self.size = data.size()

    def samples(self, rays: torch.Tensor, blur_sigma: Union[torch.Tensor, None]=None) -> torch.Tensor:
        """
        Interpolates samples from the stored image where the given rays intersect the y-axis.
        :param rays: tensor of rays
        :param blur_sigma: (optional) sigma with which to apply a Gaussian blur to the image before sampling
        :return: tensor of samples for each ray
        """

        data = self.data if blur_sigma is None else tools.gaussian_blur1d(self.data, blur_sigma.item())
        xs = rays[:, 1] - (rays[:, 0] / rays[:, 2]) * rays[:, 3]
        xs_transformed = .5 * (xs + 1.) * (torch.tensor(self.size, dtype=torch.float32) - 1.)
        i0s = torch.floor(xs_transformed.clone().detach()).type(torch.int64)
        fs = xs_transformed - i0s.type(torch.float32)
        with_zero = torch.cat((torch.zeros(1), data))
        i0s = i0s + 1
        i1s = i0s + 1
        n = with_zero.size()[0]
        i0s[i0s < 1] = 0
        i0s[i0s >= n] = 0
        i1s[i1s < 1] = 0
        i1s[i1s >= n] = 0
        return (1. - fs) * with_zero.gather(0, i0s) + fs * with_zero.gather(0, i1s)

    def display(self, axes):
        axes.pcolormesh(self.data[None, :], cmap='gray')