import numpy as np
import torch
from typing import Tuple

from torch.xpu import device


def rotate_vector2d(vector: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    return torch.matmul(torch.tensor([[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]]), vector)


def cross_vectors2d(vectors: torch.Tensor) -> torch.Tensor:
    return torch.cat((-vectors[:, 1, None], vectors[:, 0, None]), 1)


def fix_angle(angle: torch.Tensor) -> torch.Tensor:
    """
    :param angle:
    :return: the given angle expressed between -pi and pi
    """
    return torch.fmod(angle + torch.pi, 2. * torch.pi) - torch.pi


def gaussian_blur1d(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    :param image:
    :param sigma: blur factor
    :return: the given 1D image blurred with a Gaussian kernel of the given standard deviation
    """
    kernel_size = int(np.ceil(2. * sigma))
    indices = torch.arange(-kernel_size, kernel_size + 1, dtype=torch.float32)
    kernel = torch.nn.functional.normalize(torch.exp(-(indices / sigma).square()), p=1., dim=0)
    return torch.nn.functional.conv1d(image[None, None, :], kernel[None, None, :], padding='same')[0, 0]


def grid_sample1d(data: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    positions_transformed = .5 * (positions + 1.) * (torch.tensor(data.size(), dtype=torch.float32) - 1.)
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
    ret = (1. - fs) * with_zero[i0s] + fs * with_zero[i1s]

    return ret, weights


def grid_sample3d(data: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    positions_transformed = .5 * (positions + 1.) * (torch.tensor(data.size(), dtype=torch.float32, device=data.device) - 1.)
    i0s = torch.floor(positions_transformed.clone().detach()).type(torch.int64)
    fs = positions_transformed - i0s.type(torch.float32)
    with_zero = torch.cat((torch.zeros(data.size()[1], data.size()[2], device=data.device)[None, :, :], data))
    i0s[:, 0] = i0s[:, 0] + 1
    i1s = i0s + 1
    i0s_out = torch.logical_not(torch.logical_and((i0s >= torch.tensor([1, 0, 0], device=data.device)).prod(dim=-1), (i0s < torch.tensor(with_zero.size(), device=data.device)).prod(dim=-1)))
    i1s_out = torch.logical_not(torch.logical_and((i1s >= torch.tensor([1, 0, 0], device=data.device)).prod(dim=-1), (i1s < torch.tensor(with_zero.size(), device=data.device)).prod(dim=-1)))
    i0s[i0s_out, 0] = 0
    i1s[i1s_out, 0] = 0

    # determining image-edge weight modifications
    # weights = (1. - fs) * torch.logical_not(i0s_out).type(torch.float32) + fs * torch.logical_not(i1s_out).type(torch.float32)

    # sampling
    is_x0 = i0s[:, 0, None].repeat(1, 4)
    js_x = torch.cat((i0s[:, 1, None].repeat(1, 2), i1s[:, 1, None].repeat(1, 2)), dim=-1)
    ks_x = torch.cat((i0s[:, 2, None], i1s[:, 2, None], i0s[:, 2, None], i1s[:, 2, None]), dim=-1)
    is_x1 = i1s[:, 0, None].repeat(1, 4)
    x_interpolated = (1. - fs[:, 0, None]) * with_zero[is_x0, js_x, ks_x] + fs[:, 0, None] * with_zero[is_x1, js_x, ks_x]
    xy_interpolated = (1. - fs[:, 1, None]) * x_interpolated[:, 0:2] + fs[:, 1, None] * x_interpolated[:, 2:4]
    xyz_interpolated = (1. - fs[:, 2]) * xy_interpolated[:, 0] + fs[:, 2] * xy_interpolated[:, 1]
    return xyz_interpolated

    #ret = (1. - fs) * with_zero[i0s[:, 0], i0s[:, 1], i0s[:, 2]] + fs * with_zero[i1s[:, 0], i1s[:, 1], i1s[:, 2]]

    #return ret #, weights


def weighted_zero_normalised_cross_correlation(xs: torch.Tensor, ys: torch.Tensor, ns: torch.Tensor):
    """
    :param xs: a tensor of values
    :param ys: a tensor of values
    :param ns: a tensor of weights for the value pairs, each between 0 and 1
    :return: The zero-normalised cross correlation between `xs` and `ys`, each pair weighted in contribution by the
             weight given in `ns`
    """
    sum_n = ns.sum_to_size(1)
    sum_x = (ns * xs).sum_to_size(1)
    sum_y = (ns * ys).sum_to_size(1)
    sum_x2 = (ns * xs.square()).sum_to_size(1)
    sum_y2 = (ns * ys.square()).sum_to_size(1)
    sum_prod = (xs * ys * ns).sum_to_size(1)
    numerator = sum_n * sum_prod - sum_x * sum_y
    denominator = torch.sqrt(sum_n * sum_x2 - sum_x * sum_x) * torch.sqrt(sum_n * sum_y2 - sum_y * sum_y)
    return numerator / denominator