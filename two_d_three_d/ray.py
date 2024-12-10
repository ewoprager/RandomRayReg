from typing import Union, Tuple, TypeAlias
import torch

from transformation import SE3 as Transformation


class Ray:
    """
    A ray is stored as a row of a 6-column tensor. The first 3 values are a position in CT space through which the ray
    passes, and the final 3 values are a unit-length direction vector of the ray.
    """
    Transformation: TypeAlias = Transformation

    @staticmethod
    def transform(rays: torch.Tensor, theta: Transformation) -> torch.Tensor:
        t = theta.get_matrix().t().to(rays.device)
        t2 = torch.cat((torch.cat((t, torch.zeros_like(t))), torch.cat((torch.zeros_like(t), t))), dim=1)
        ray_count = rays.size()[0]
        rays_homogeneous = torch.cat((rays[:, 0:3], torch.ones(ray_count, device=rays.device)[:, None], rays[:, 3:6], torch.zeros(ray_count, device=rays.device)[:, None]), dim=1)
        ret_homogeneous = torch.matmul(rays_homogeneous, t2)
        return ret_homogeneous[:, torch.tensor([0, 1, 2, 4, 5, 6])]

    @staticmethod
    def xy_plane_intersections(rays: torch.Tensor, z_offset: Union[float, None]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param rays: tensor rays
        :param z_offset: offset of the plane from the x-y plane
        :return: Tensor of the (x, y) positions at which the rays intersect the x-y plane, the values of lambda at the intersections
        """
        lambdas = - ((rays[:, 2] if z_offset is None else rays[:, 2] - z_offset) / rays[:, 5])
        return rays[:, 0:2] + lambdas[:, None] * rays[:, 3:5], lambdas

    @staticmethod
    def yz_plane_intersections(rays: torch.Tensor, x_offset: Union[float, None]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param rays: tensor rays
        :param x_offset: offset of the plane from the y-z plane
        :return: Tensor of the (y, z) positions at which the rays intersect the y-z plane, the values of lambda at the intersections
        """
        lambdas = - ((rays[:, 0] if x_offset is None else rays[:, 0] - x_offset) / rays[:, 3])
        return rays[:, 1:3] + lambdas[:, None] * rays[:, 4:6], lambdas

    @staticmethod
    def xz_plane_intersections(rays: torch.Tensor, y_offset: Union[float, None]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param rays: tensor rays
        :param y_offset: offset of the plane from the x-z plane
        :return: Tensor of the (x, z) positions at which the rays intersect the x-z plane, the values of lambda at the intersections
        """
        xz = torch.tensor([0, 2])
        xdzd = torch.tensor([3, 5])
        lambdas = - ((rays[:, 1] if y_offset is None else rays[:, 1] - y_offset) / rays[:, 4])
        return rays[:, xz] + lambdas[:, None] * rays[:, xdzd], lambdas

    @staticmethod
    def scores(rays, source_position: torch.Tensor, alpha: torch.Tensor=torch.tensor([0.14])) -> torch.Tensor:
        """
        Find the scores of the given rays.

        A ray is scored according to the following formula:

        `score = exp(-alpha * distance^2)`

        where `alpha` is the given parameter, and `distance` is the shortest distance from the ray to the given source
        position.
        :param rays: tensor of rays
        :param source_position: position of simulated X-ray source
        :param alpha: drop-off coefficient for ray score `vs.` distance
        :return: tensor of ray scores
        """
        origin_offsets = rays[:, 0:3] - source_position
        scaled_distances = torch.norm(origin_offsets - (origin_offsets * rays[:, 3:6]).sum(dim=-1)[:, None] * rays[:, 3:6], dim=1) / alpha
        return torch.exp(- scaled_distances * scaled_distances)

    @staticmethod
    def generate_random(count: int) -> torch.Tensor:
        rands = -1. + 2. * torch.rand(count, 6)
        return torch.cat((rands[:, 0:3], torch.nn.functional.normalize(rands[:, 3:6], dim=1)), dim=1)

    @staticmethod
    def generate_true_untransformed(size: torch.Tensor,
                                    source_position: torch.Tensor,
                                    *,
                                    device) -> torch.Tensor:
        """
        :param size: 2D image size
        :param source_position: position of simulated X-ray source
        :return: tensor of rays that would correspond to a `width` x `height` pixel DRR sitting on the x-y plane
        """
        source_position_device = source_position.to(device)
        y_values, x_values = torch.meshgrid(torch.linspace(-1., 1., size[0].item(), device=device), torch.linspace(-1., 1., size[1].item(), device=device))
        plane_intersections = torch.cat((y_values.flatten()[:, None], x_values.flatten()[:, None]), dim=1)
        count = (size[0] * size[1]).item()
        return torch.cat((source_position_device.repeat((count, 1)), torch.nn.functional.normalize(torch.cat((plane_intersections, torch.zeros(count, 1, device=device)), dim=1) - source_position_device, dim=1)), dim=1)


    # def plot(axes, rays: torch.Tensor, shades: torch.Tensor):
    #     for i in range(rays.size()[0].item()):
    #         r: float = shades[i].item()
    #         row = rays[i]
    #         axes.plot([-1.5, 1.5], [row[1] - row[3] * (row[0] + 1.5) / row[2], row[1] - row[3] * (row[0] - 1.5) / row[2]], color=(r, r, r))

