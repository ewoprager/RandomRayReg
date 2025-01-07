from typing import Union, Tuple
import numpy as np
import torch
import kornia
import nrrd

import tools
from two_d_three_d.ray import Ray
import debug


class Volume:
    """
    A 3D image that rays can be integrated along through
    """
    __init_key = object()

    def __init__(self, init_key, path: str, data: torch.Tensor):
        """
        Private constructor
        :param init_key:
        :param data: tensor of raw CT data in Houndsfield units
        """
        assert (init_key is self.__class__.__init_key), "Constructor is private"
        self.path = path

        data = torch.maximum(data.type(torch.float32) + 1000., torch.tensor([0.], device=data.device))
        self.data: list[torch.Tensor] = [data]

        debug.tic("Down-sampling {} x {} x {} volume".format(data.size()[0], data.size()[1], data.size()[2]))
        down_sampler = torch.nn.AvgPool3d(2)
        while (torch.tensor(self.data[-1].size()) // 2 > 0).all():
            self.data.append(down_sampler(self.data[-1][None, :, :, :])[0])
        debug.toc("Generated {} volumes".format(len(self.data) - 1))

    @classmethod
    def from_file(cls, path: str, *, device):
        data, _ = nrrd.read(path)
        return cls(cls.__init_key, path, torch.tensor(data, device=device))

    @classmethod
    def load(cls, cache_directory: str, *, device):
        path = torch.load(cache_directory + "/volume.pt")
        return cls.from_file(path, device=device)

    def save(self, cache_directory: str):
        torch.save(self.path, cache_directory + "/volume.pt")

    def samples(self, positions: torch.Tensor, *, mip_level: int = 0) -> torch.Tensor:
        """
        :param positions: a vector of 3D vectors, positions between (-1,-1) and (1,1)
        :return: vector of bi-linearly interpolated samples from the stored image at the given positions
        """

        # return tools.grid_sample3d(self.data[mip_level], positions)

        # data_cpu = self.data[None, None, :, :, :].cpu()
        # positions_cpu = positions[None, None, None, :, :].cpu()
        # torch.nn.functional.grid_sample does not yet work on `mps` device, but tried implementing my own version in
        # Python that could use `mps`, but it was slower.
        vol_data = self.data[mip_level]
        return torch.nn.functional.grid_sample(vol_data[None, None, :, :, :], positions[None, None, None, :, :],
            align_corners=False)[0, 0, 0, 0]

    def integrate(self, rays: torch.Tensor, *, alpha: float, n: Union[int, None] = None,
                  mip_level: int = 0) -> torch.Tensor:
        """
        :param rays: tensor of rays to integrate along
        :param n: The number of points to sample along each ray
        :param alpha: X-ray attenuation factor
        :return: A tensor of approximations of the X-ray intensities attenuated along the given rays through the CT
                 volume. This is calculated as `1 - exp(-alpha * sum)` where `sum` is the approximate average value
                 along rays in the CT volume.
        """
        if n is None:
            n = torch.tensor(self.data[mip_level].size()).max().item()

        inters_x0, ls_x0 = Ray.yz_plane_intersections(rays, -1.)
        inters_x1, ls_x1 = Ray.yz_plane_intersections(rays, 1.)
        inters_y0, ls_y0 = Ray.xz_plane_intersections(rays, -1.)
        inters_y1, ls_y1 = Ray.xz_plane_intersections(rays, 1.)
        inters_z0, ls_z0 = Ray.xy_plane_intersections(rays, -1.)
        inters_z1, ls_z1 = Ray.xy_plane_intersections(rays, 1.)

        inters_catted = torch.cat((
        inters_x0[None, :], inters_x1[None, :], inters_y0[None, :], inters_y1[None, :], inters_z0[None, :],
        inters_z1[None, :]), dim=0)
        within = torch.logical_and((inters_catted > -1.).prod(dim=2), (inters_catted < 1.).prod(dim=2))
        lambdas_catted = torch.cat(
            (ls_x0[None, :], ls_x1[None, :], ls_y0[None, :], ls_y1[None, :], ls_z0[None, :], ls_z1[None, :]))
        start_lambdas = lambdas_catted.where(within, torch.inf).min(dim=0)[0]
        end_lambdas = lambdas_catted.where(within, -torch.inf).max(dim=0)[0]
        deltas = ((end_lambdas - start_lambdas) / float(n))[:, None] * rays[:, 3:6]

        ps = rays[:, 0:3] + start_lambdas[:, None] * rays[:, 3:6]
        ret = torch.zeros(rays.size()[0], device=self.data[mip_level].device)
        for _ in range(n):
            ret += self.samples(ps, mip_level=mip_level)
            ps += deltas
        ret = 1. - torch.exp(-ret / (alpha * float(n)))
        ret[ret.isnan()] = 0.
        return ret

    # def display(self, axes):  #     X, Y = np.meshgrid(np.linspace(-1., 1., self.size[0], endpoint=True), np.linspace(-1., 1., self.size[1], endpoint=True))  #     axes.pcolormesh(X, Y, self.data, cmap='gray')


class Image:
    """
    A 2D image that can be sampled using rays
    """

    def __init__(self, data: torch.Tensor, *, generate_mip_levels: bool = False):
        self.data: list[torch.Tensor] = [data]

        if generate_mip_levels:
            debug.tic("Down-sampling {} x {} image".format(data.size()[0], data.size()[1]))
            down_sampler = torch.nn.AvgPool2d(2)
            while (torch.tensor(self.data[-1].size()) // 2 > 0).all():
                self.data.append(down_sampler(self.data[-1][None, :, :])[0])
            debug.toc("Generated {} images".format(len(self.data) - 1))

    def samples(self, rays: torch.Tensor, *, blur_sigma: Union[torch.Tensor, None] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Interpolates samples from the stored image where the given rays intersect the y-axis.
        :param rays: tensor of rays
        :param blur_sigma: (optional) sigma with which to apply a Gaussian blur to the image before sampling
        :return: tensor of samples for each ray, tensor of weight modifications for each ray
        """
        data = self.data[0] if blur_sigma is None else \
        kornia.filters.gaussian_blur2d(self.data[0][None, None, :, :], 1 + 2 * int(np.ceil(2. * blur_sigma.item())),
            blur_sigma.repeat(1, 2))[0, 0]
        positions, _ = Ray.xy_plane_intersections(rays)

        # sampling

        # !!! torch.nn.functional.grid_sample can SIGSEGV with too large a number of positions; own version does not and
        # doesn't seem to run any slower on cpu.
        # ret = torch.nn.functional.grid_sample(data[None, None, :, :], positions[None, None, :, :], align_corners=False)[0, 0, 0]
        ret, weights = tools.grid_sample2d(data, positions)

        # determining image-edge weight modifications
        # weights = (1. - fs) * torch.logical_not(i0s_out).type(torch.float32) + fs * torch.logical_not(i1s_out).type(torch.float32)

        return ret, weights, positions

    def display(self, axes, *, mip_level: int = 0):
        if mip_level >= len(self.data):
            print("Requested mip level {} out of {}; displaying level {}".format(mip_level, len(self.data) - 1,
                len(self.data) - 1))
            mip_level = len(self.data) - 1

        xs, ys = np.meshgrid(np.linspace(-1., 1., self.data[mip_level].size()[0], endpoint=True),
            np.linspace(-1., 1., self.data[mip_level].size()[1], endpoint=True))
        axes.pcolormesh(xs, ys, self.data[mip_level].cpu(), cmap='gray')
