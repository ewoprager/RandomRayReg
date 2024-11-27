import numpy as np
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass


def rotate_vector(vector: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    return torch.matmul(torch.tensor([[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]]), vector)


def cross_vectors(vectors: torch.Tensor) -> torch.Tensor:
    return torch.cat((-vectors[:, 1, None], vectors[:, 0, None]), 1)


def transform_rays(_rays: torch.Tensor, _theta: torch.Tensor) -> torch.Tensor:
    transform = torch.cat((torch.cat((torch.cos(_theta), torch.sin(_theta)))[None, :], torch.cat((-torch.sin(_theta), torch.cos(_theta)))[None, :]))
    transform2 = torch.cat((torch.cat((transform, torch.zeros_like(transform))), torch.cat((torch.zeros_like(transform), transform))), dim=1)
    return torch.matmul(_rays, transform2)


def ray_scores(_rays, source: torch.Tensor, alpha: torch.Tensor=torch.tensor([50.0])) -> torch.Tensor:
    signed_distances = ((_rays[:, 0:2] - source) * cross_vectors(_rays[:, 2:4])).sum(dim=1)
    return torch.exp(-alpha * signed_distances * signed_distances)


def random_rays(count: int) -> torch.Tensor:
    rands = -1. + 2. * torch.rand(count, 4)
    return torch.cat((rands[:, 0:2], torch.nn.functional.normalize(rands[:, 2:4], dim=1)), dim=1)


def true_rays(count: int) -> torch.Tensor:
    return torch.cat((torch.zeros(count, 1), torch.linspace(-1., 1., count)[:, None], torch.ones(count, 1), torch.zeros(count, 1)), dim=1)


@dataclass
class Volume:
    data: torch.Tensor

    def samples(self, positions: torch.Tensor) -> torch.Tensor:
        """
        :param positions: a vector of 2D vectors, positions between (-1,-1) and (1,1)
        :return: vector of bi-linear interpolated samples from stored image at given positions
        """
        return torch.nn.functional.grid_sample(self.data[None, None, :, :], positions[None, None, :, :], align_corners=False)

    def integrate(self, _rays: torch.Tensor, n=20) -> torch.Tensor:
        perps = cross_vectors(_rays[:, 2:4])
        offsets = (_rays[:, 0:2] * perps).sum(dim=1)[:, None]
        ps = -np.sqrt(2.) * _rays[:, 2:4] + offsets * perps
        deltas = (2. * np.sqrt(2.) / float(n)) * _rays[:, 2:4]
        ret = torch.zeros(_rays.size()[0])
        for i in range(n):
            ret += self.samples(ps)[0, 0, 0]
            ps += deltas
        return 1. - torch.exp(-0.1 * ret)

    def display(self, axes):
        X, Y = np.meshgrid(np.linspace(-1., 1., self.data.size()[0], endpoint=True), np.linspace(-1., 1., self.data.size()[1], endpoint=True))
        axes.pcolormesh(X, Y, self.data, cmap='gray')


@dataclass
class Image:
    image: torch.Tensor

    def samples(self, _rays: torch.Tensor) -> torch.Tensor:
        xs = _rays[:, 1] - (_rays[:, 0] / _rays[:, 2]) * _rays[:, 3]
        xs_transformed = .5 * (xs + 1.) * (torch.tensor(self.image.size(), dtype=torch.float32) - 1.)
        i0s = torch.floor(xs_transformed.clone().detach()).type(torch.int64)
        fs = xs_transformed - i0s.type(torch.float32)
        with_zero = torch.cat((torch.zeros(1), self.image))
        i0s = i0s + 1
        i1s = i0s + 1
        n = with_zero.size()[0]
        i0s[i0s < 1] = 0
        i0s[i0s >= n] = 0
        i1s[i1s < 1] = 0
        i1s[i1s >= n] = 0
        return (1. - fs) * with_zero.gather(0, i0s) + fs * with_zero.gather(0, i1s)

    def display(self, axes):
        axes.pcolormesh(self.image[None, :], cmap='gray')


class CrossCorrelation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs, ys, ns):
        sum_n = ns.sum_to_size(1)
        sum_x = (ns * xs).sum_to_size(1)
        sum_y = (ns * ys).sum_to_size(1)
        sum_x2 = (ns * xs.square()).sum_to_size(1)
        sum_y2 = (ns * ys.square()).sum_to_size(1)
        sum_prod = (xs * ys * ns).sum_to_size(1)
        numerator = sum_n * sum_prod - sum_x * sum_y
        denominator = torch.sqrt(sum_n * sum_x2 - sum_x * sum_x) * torch.sqrt(sum_n * sum_y2 - sum_y * sum_y)
        return numerator / denominator


if __name__ == "__main__":
    ray_count: int = 10000
    source_position: torch.Tensor = torch.tensor([5.0, 0.0])
    image_size: int = 10
    volume_size: int = 6

    # CT space is 2D, exists between (-1,-1) and (1,1)
    volume_data = torch.rand(volume_size, volume_size)
    volume = Volume(volume_data)

    # generate image at random theta
    theta_true = torch.pi * (-.5 + torch.rand(1))
    print("Theta*", theta_true)
    rays_true = transform_rays(true_rays(image_size), theta_true)
    image_data = volume.integrate(rays_true)
    image = Image(image_data)
    _, ax0 = plt.subplots()
    image.display(ax0)
    plt.show()

    # rays are defined by two 2D vectors: an origin and direction in CT space
    rays = random_rays(ray_count)
    print("Rays:", rays)

    samples0 = image.samples(rays)
    print("Samples at zero transform: ", samples0)

    # display volume on its own
    _, ax1 = plt.subplots()
    volume.display(ax1)
    ax1.set_ylim(-1.5, 1.5)
    plt.show()

    # display volume, with rays colours according to fixed image brightness
    _, ax1 = plt.subplots()
    volume.display(ax1)
    for i in range(rays.size()[0]):
        r = float(samples0[i].item())
        row = rays[i]
        ax1.plot([-1.5, 1.5], [row[1] - row[3] * (row[0] + 1.5) / row[2], row[1] - row[3] * (row[0] - 1.5) / row[2]], color=(r, r, r))
    ax1.set_ylim(-1.5, 1.5)
    plt.show()

    # display volume, with rays coloured according to volume integral
    intensities = volume.integrate(rays)
    print("Intensities at zero transform: ", intensities)
    min_intensity: float = intensities.min().item()
    intensity_range = float(intensities.max().item()) - min_intensity
    _, ax2 = plt.subplots()
    volume.display(ax2)
    for i in range(rays.size()[0]):
        r = (float(intensities[i].item()) - min_intensity) / intensity_range
        row = rays[i]
        ax2.plot([-1.5, 1.5], [row[1] - row[3] * (row[0] + 1.5) / row[2], row[1] - row[3] * (row[0] - 1.5) / row[2]], color=(r, r, r))
    ax2.set_ylim(-1.5, 1.5)
    plt.show()

    print("Scores at zero transform: ", ray_scores(rays, source_position))

    # rays.requires_grad_(True)
    # m = 30
    # for i in range(m):
    #     theta = torch.tensor([float(i) * 2. * torch.pi / float(m)])
    #     theta.requires_grad_(True)
    #     transformed_rays = transform_rays(rays, theta)
    #     plt.pcolormesh(volume.integrate(transform_rays(image.true_rays(), theta.clone().detach()))[None, :],
    #                    cmap='gray')
    #     plt.show()
    #     scores = ray_scores(transformed_rays, source_position)
    #     samples = image.samples(transformed_rays)
    #     cross_correlation = CrossCorrelation()
    #     s = cross_correlation(samples, intensities, scores)
    #     # theta.retain_grad()
    #     s.backward(torch.ones_like(s))
    #     grad = theta.grad
    #     theta.requires_grad_(False)
    #     print(grad)

    # lr: float = 0.5
    # lr1: float = 0.01
    # m: int = 50
    # r: float = pow(lr1 / lr, 1. / float(m))
    # for i in range(50):
    #     theta.grad = torch.zeros_like(theta)
    #     theta.requires_grad_(True)
    #     transformed_rays = transform_rays(rays, theta)
    #     plt.pcolormesh(volume.integrate(transform_rays(true_rays(image_size), theta.clone().detach()))[None, :],
    #                    cmap='gray')
    #     plt.show()
    #     scores = ray_scores(transformed_rays, source_position)
    #     samples = image.samples(transformed_rays)
    #     cross_correlation = CrossCorrelation()
    #     s = cross_correlation(samples, intensities, scores)
    #     s.backward(torch.ones_like(s))
    #     grad = theta.grad
    #     theta.requires_grad_(False)
    #     print(theta, grad)
    #     if grad * grad < 1e-6:
    #         break
    #     theta += lr * grad
    #     lr *= r

    for j in range(5):
        theta = torch.pi * (-.5 + torch.rand(1))
        optimiser = torch.optim.ASGD([theta], lr=.1, maximize=True)
        for i in range(50):
            optimiser.zero_grad()
            theta.grad = torch.zeros_like(theta)
            theta.requires_grad_(True)
            transformed_rays = transform_rays(rays, theta)
            plt.pcolormesh(volume.integrate(transform_rays(true_rays(image_size), theta.clone().detach()))[None, :],
                           cmap='gray')
            plt.show()
            scores = ray_scores(transformed_rays, source_position)
            samples = image.samples(transformed_rays)
            cross_correlation = CrossCorrelation()
            s = cross_correlation(samples, intensities, scores)
            s.backward(torch.ones_like(s))
            grad = theta.grad
            theta.requires_grad_(False)
            # print(theta, grad)
            optimiser.step()
        print(theta)