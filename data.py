import numpy as np
import torch

import tools

class Volume:
    def __init__(self, data: torch.Tensor):
        self.data = data
        self.size = data.size()

    def samples(self, positions: torch.Tensor) -> torch.Tensor:
        """
        :param positions: a vector of 2D vectors, positions between (-1,-1) and (1,1)
        :return: vector of bi-linear interpolated samples from stored image at given positions
        """
        return torch.nn.functional.grid_sample(self.data[None, None, :, :], positions[None, None, :, :], align_corners=False)

    def integrate(self, _rays: torch.Tensor, n: int=20, alpha: float=0.1) -> torch.Tensor:
        perps = tools.cross_vectors(_rays[:, 2:4])
        offsets = (_rays[:, 0:2] * perps).sum(dim=1)[:, None]
        ps = -np.sqrt(2.) * _rays[:, 2:4] + offsets * perps
        deltas = (2. * np.sqrt(2.) / float(n)) * _rays[:, 2:4]
        ret = torch.zeros(_rays.size()[0])
        for i in range(n):
            ret += self.samples(ps)[0, 0, 0]
            ps += deltas
        return 1. - torch.exp(-alpha * ret)

    def display(self, axes):
        X, Y = np.meshgrid(np.linspace(-1., 1., self.size[0], endpoint=True), np.linspace(-1., 1., self.size[1], endpoint=True))
        axes.pcolormesh(X, Y, self.data, cmap='gray')


class Image:
    def __init__(self, data: torch.Tensor):
        self.data = data
        self.size = data.size()

    def samples(self, _rays: torch.Tensor) -> torch.Tensor:
        xs = _rays[:, 1] - (_rays[:, 0] / _rays[:, 2]) * _rays[:, 3]
        xs_transformed = .5 * (xs + 1.) * (torch.tensor(self.size, dtype=torch.float32) - 1.)
        i0s = torch.floor(xs_transformed.clone().detach()).type(torch.int64)
        fs = xs_transformed - i0s.type(torch.float32)
        with_zero = torch.cat((torch.zeros(1), self.data))
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