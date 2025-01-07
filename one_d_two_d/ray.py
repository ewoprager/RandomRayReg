from typing import TypeAlias
import torch

import tools
from transformation import Rotation1D as Transformation


class Ray:
    """
    A ray is stored as a row of a 4-column tensor. The first 2 values are a position in CT space through which the ray
    passes, and the final 2 values are a unit-length direction vector of the ray.
    """
    Transformation: TypeAlias = Transformation

    @staticmethod
    def transform(rays: torch.Tensor, theta: Transformation) -> torch.Tensor:
        t = theta.get_matrix().t()
        t2 = torch.cat((torch.cat((t, torch.zeros_like(t))), torch.cat((torch.zeros_like(t), t))), dim=1)
        return torch.matmul(rays, t2)

    @staticmethod
    def y_axis_intersections(rays: torch.Tensor) -> torch.Tensor:
        """
        :param rays: tensor rays
        :return: Tensor of the x-positions at which the rays intersect the y-axis
        """
        return rays[:, 1] - (rays[:, 0] / rays[:, 2]) * rays[:, 3]

    @staticmethod
    def scores(rays, source_position: torch.Tensor, alpha: torch.Tensor = torch.tensor([0.14])) -> torch.Tensor:
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
        scaled_signed_distances = ((rays[:, 0:2] - source_position) * tools.cross_vectors2d(rays[:, 2:4])).sum(
            dim=1) / alpha
        return torch.exp(- scaled_signed_distances * scaled_signed_distances)

    @staticmethod
    def generate_random(count: int) -> torch.Tensor:
        rands = -1. + 2. * torch.rand(count, 4)
        return torch.cat((rands[:, 0:2], torch.nn.functional.normalize(rands[:, 2:4], dim=1)), dim=1)

    @staticmethod
    def generate_true_untransformed(count: torch.Tensor, source_position: torch.Tensor, *, device) -> torch.Tensor:
        """
        :param count:
        :param source_position: position of simulated X-ray source
        :return: tensor of rays that would correspond to a `count` pixel DRR sitting on the y-axis
        """
        source_position_device = source_position.to(device)
        y_plane_intersections = torch.cat((
        torch.zeros(count.item(), 1, device=device), torch.linspace(-1., 1., count.item(), device=device)[:, None]),
            dim=1)
        return torch.cat((source_position_device.repeat((count.item(), 1)),
        torch.nn.functional.normalize(y_plane_intersections - source_position_device, dim=1)), dim=1)

    @staticmethod
    def plot(axes, rays: torch.Tensor, shades: torch.Tensor):
        for i in range(rays.size()[0].item()):
            r: float = shades[i].item()
            row = rays[i]
            axes.plot([-1.5, 1.5],
                [row[1] - row[3] * (row[0] + 1.5) / row[2], row[1] - row[3] * (row[0] - 1.5) / row[2]], color=(r, r, r))
