from typing import Union, Tuple
import numpy as np
import torch
import kornia
import time

import tools
from two_d_three_d.ray import Ray

class Volume:
    """
    A 3D image that rays can be integrated along through
    """
    def __init__(self, data: torch.Tensor):
        self.data = data
        self.size = data.size()

    def samples(self, positions: torch.Tensor) -> torch.Tensor:
        """
        :param positions: a vector of 3D vectors, positions between (-1,-1) and (1,1)
        :return: vector of bi-linearly interpolated samples from the stored image at the given positions
        """

        # return tools.grid_sample3d(self.data, positions)

        # data_cpu = self.data[None, None, :, :, :].cpu()
        # positions_cpu = positions[None, None, None, :, :].cpu()
        return torch.nn.functional.grid_sample(self.data[None, None, :, :, :], positions[None, None, None, :, :], align_corners=False)[0, 0, 0, 0].to(self.data.device)

    def integrate(self, rays: torch.Tensor, n: int=500, alpha: float=.5) -> torch.Tensor:
        """
        :param rays: tensor of rays to integrate along
        :param n: The number of points to sample along each ray
        :param alpha: X-ray attenuation factor
        :return: A tensor of approximations of the X-ray intensities attenuated along the given rays through the CT
                 volume. This is calculated as `1 - exp(-alpha * sum)` where `sum` is the approximate average value
                 along rays in the CT volume.
        """
        print("Integrating {} rays".format(rays.size()[0]))
        print("Pre-processing...")
        tic = time.time()

        inters_x0, ls_x0 = Ray.yz_plane_intersections(rays, -1.)
        inters_x1, ls_x1 = Ray.yz_plane_intersections(rays, 1.)
        inters_y0, ls_y0 = Ray.xz_plane_intersections(rays, -1.)
        inters_y1, ls_y1 = Ray.xz_plane_intersections(rays, 1.)
        inters_z0, ls_z0 = Ray.xy_plane_intersections(rays, -1.)
        inters_z1, ls_z1 = Ray.xy_plane_intersections(rays, 1.)

        inters_catted = torch.cat((inters_x0[None, :], inters_x1[None, :], inters_y0[None, :], inters_y1[None, :], inters_z0[None, :], inters_z1[None, :]), dim=0)
        within = torch.logical_and((inters_catted > -1.).prod(dim=2), (inters_catted < 1.).prod(dim=2))
        lambdas_catted = torch.cat((ls_x0[None, :], ls_x1[None, :], ls_y0[None, :], ls_y1[None, :], ls_z0[None, :], ls_z1[None, :]))
        start_lambdas = lambdas_catted.where(within, torch.inf).min(dim=0)[0]
        end_lambdas = lambdas_catted.where(within, -torch.inf).max(dim=0)[0]
        deltas = ((end_lambdas - start_lambdas) / float(n))[:, None] * rays[:, 3:6]

        ps = rays[:, 0:3] + start_lambdas[:, None] * rays[:, 3:6]
        ret = torch.zeros(rays.size()[0], device=self.data.device)
        toc = time.time()
        print("Done. Took {:.3f}s".format(toc - tic))
        print("Looping...")
        tic = time.time()
        for i in range(n):
            ret += self.samples(ps)
            ps += deltas
        toc = time.time()
        print("Done. Took {:.3f}s".format(toc - tic))
        return 1. - torch.exp(-ret / (alpha * float(n)))

    # def display(self, axes):
    #     X, Y = np.meshgrid(np.linspace(-1., 1., self.size[0], endpoint=True), np.linspace(-1., 1., self.size[1], endpoint=True))
    #     axes.pcolormesh(X, Y, self.data, cmap='gray')


class Image:
    """
    A 2D image that can be sampled using rays
    """
    def __init__(self, data: torch.Tensor):
        self.data = data
        self.size = data.size()

    def samples(self, rays: torch.Tensor, blur_sigma: Union[torch.Tensor, None]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolates samples from the stored image where the given rays intersect the y-axis.
        :param rays: tensor of rays
        :param blur_sigma: (optional) sigma with which to apply a Gaussian blur to the image before sampling
        :return: tensor of samples for each ray, tensor of weight modifications for each ray
        """
        data = self.data if blur_sigma is None else kornia.filters.gaussian_blur2d(self.data[None, None, :, :], 1 + 2 * int(np.ceil(2. * blur_sigma.item())), blur_sigma.repeat(1, 2))[0, 0]
        positions, _ = Ray.xy_plane_intersections(rays)

        # sampling
        ret = torch.nn.functional.grid_sample(self.data[None, None, :, :], positions[None, None, :, :], align_corners=False)[0, 0, 0]

        # determining image-edge weight modifications
        #weights = (1. - fs) * torch.logical_not(i0s_out).type(torch.float32) + fs * torch.logical_not(i1s_out).type(torch.float32)

        return ret, torch.zeros(1) #weights

    def display(self, axes):
        xs, ys = np.meshgrid(np.linspace(-1., 1., self.size[0], endpoint=True), np.linspace(-1., 1., self.size[1], endpoint=True))
        axes.pcolormesh(xs, ys, self.data.cpu(), cmap='gray')

    def save(self, path: str):
        torch.save(self.data, path)