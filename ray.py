import torch

import tools

def transform(_rays: torch.Tensor, _theta: torch.Tensor) -> torch.Tensor:
    t = torch.cat((torch.cat((torch.cos(_theta), torch.sin(_theta)))[None, :], torch.cat((-torch.sin(_theta), torch.cos(_theta)))[None, :]))
    t2 = torch.cat((torch.cat((t, torch.zeros_like(t))), torch.cat((torch.zeros_like(t), t))), dim=1)
    return torch.matmul(_rays, t2)


def scores(_rays, source_position: torch.Tensor, alpha: torch.Tensor=torch.tensor([0.14])) -> torch.Tensor:
    scaled_signed_distances = ((_rays[:, 0:2] - source_position) * tools.cross_vectors(_rays[:, 2:4])).sum(dim=1) / alpha
    return torch.exp(-scaled_signed_distances * scaled_signed_distances)


def generate_random(count: int) -> torch.Tensor:
    rands = -1. + 2. * torch.rand(count, 4)
    return torch.cat((rands[:, 0:2], torch.nn.functional.normalize(rands[:, 2:4], dim=1)), dim=1)


def generate_true_untransformed(count: int, source_position: torch.Tensor) -> torch.Tensor:
    y_plane_intersections = torch.cat((torch.zeros(count, 1), torch.linspace(-1., 1., count)[:, None]), dim=1)
    return torch.cat((source_position.repeat((count, 1)), torch.nn.functional.normalize(y_plane_intersections - source_position, dim=1)), dim=1)


def plot(axes, rays: torch.Tensor, shades: torch.Tensor):
    for i in range(rays.size()[0]):
        r: float = shades[i].item()
        row = rays[i]
        axes.plot([-1.5, 1.5], [row[1] - row[3] * (row[0] + 1.5) / row[2], row[1] - row[3] * (row[0] - 1.5) / row[2]], color=(r, r, r))

