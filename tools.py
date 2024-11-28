import numpy as np
import torch

def rotate_vector(vector: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    return torch.matmul(torch.tensor([[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]]), vector)


def cross_vectors(vectors: torch.Tensor) -> torch.Tensor:
    return torch.cat((-vectors[:, 1, None], vectors[:, 0, None]), 1)


def fix_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.fmod(angle, 2. * torch.pi)


def gaussian_blur1d(image: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_size = int(np.ceil(2. * sigma))
    indices = torch.arange(-kernel_size, kernel_size + 1, dtype=torch.float32)
    kernel = torch.nn.functional.normalize(torch.exp(-(indices / sigma).square()), p=1., dim=0)
    return torch.nn.functional.conv1d(image[None, None, :], kernel[None, None, :], padding='same')[0, 0]


def zero_normalised_cross_correlation(xs, ys, ns):
    sum_n = ns.sum_to_size(1)
    sum_x = (ns * xs).sum_to_size(1)
    sum_y = (ns * ys).sum_to_size(1)
    sum_x2 = (ns * xs.square()).sum_to_size(1)
    sum_y2 = (ns * ys.square()).sum_to_size(1)
    sum_prod = (xs * ys * ns).sum_to_size(1)
    numerator = sum_n * sum_prod - sum_x * sum_y
    denominator = torch.sqrt(sum_n * sum_x2 - sum_x * sum_x) * torch.sqrt(sum_n * sum_y2 - sum_y * sum_y)
    return numerator / denominator