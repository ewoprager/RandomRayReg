from timeit import reindent

import numpy as np
import torch
import matplotlib.pyplot as plt

import tools
import ray
import data
from tools import fix_angle
from registration import Registration
from random_rays import RandomRays


if __name__ == "__main__":
    image_size: int = 20
    volume_size: int = 10

    # CT space is 2D, exists between (-1,-1) and (1,1)
    volume_data = torch.rand(volume_size, volume_size)
    radius = 0.5 * float(volume_size - 1)
    for i in range(volume_size):
        for j in range(volume_size):
            di = float(i) - radius
            dj = float(j) - radius
            if di * di + dj * dj > radius * radius:
                volume_data[i, j] = 0.
    volume = data.Volume(volume_data)

    registration = Registration(volume, image_size, source_position=torch.tensor([3., 0.]))

    # display drr target
    _, ax0 = plt.subplots()
    registration.image.display(ax0)
    plt.show()

    # display volume
    _, ax1 = plt.subplots()
    registration.volume.display(ax1)
    ax1.set_ylim(-1., 1.)
    plt.show()

    # rays = RandomRays(registration)

    # display volume, with rays colours according to fixed image brightness
    # _, ax1 = plt.subplots()
    # registration.volume.display(ax1)
    # rays.plot_with_sampled_shading(ax1)
    # ax1.set_ylim(-1., 1.)
    # plt.show()

    # display volume, with rays coloured according to volume integral
    # _, ax2 = plt.subplots()
    # registration.volume.display(ax2)
    # rays.plot_with_intensity_shading(ax2)
    # ax2.set_ylim(-1., 1.)
    # plt.show()

    m: int = 8
    _, axes = plt.subplots()
    alpha = 1.0
    ray_density = 1000.
    blur_constant = 4.
    # for j in range(m):

    # alpha = 1. / np.sqrt(0.3 * pow(1.5, m - j - 1))
    ray_count = int(np.ceil(abs(registration.source_position[0]) * ray_density * np.sqrt(np.pi) / alpha))
    rays = RandomRays(registration, ray_count=ray_count)
    thetas = np.linspace(-torch.pi, torch.pi, 200, dtype=np.float32)
    ss = np.zeros_like(thetas)
    ssn = 0.
    for i in range(len(thetas)):
        s, sn = rays.evaluate(torch.tensor([thetas[i]]), alpha=torch.tensor([alpha]), blur_constant=torch.tensor([blur_constant]))
        ss[i] = s.item()
        ssn += sn
    asn = ssn / float(len(thetas))
    r: float = float(j) / float(m - 1)
    axes.plot(thetas, ss,  label="alpha = {:.3f}, {} rays, av. sum n = {:.3f}, bc = {:.3f}".format(alpha, ray_count, asn, blur_constant))
    axes.vlines(-registration.true_theta.item(), -1., axes.get_ylim()[1])
    # ray_density *= 2.

    plt.legend()
    plt.show()

    exit()

    thetas = []
    ss = []

    theta = torch.pi * (-1. + 2. * torch.rand(1))
    optimiser = torch.optim.ASGD([theta], lr=1.1)
    for i in range(100):
        def closure():
            optimiser.zero_grad()
            thetas.append(fix_angle(theta).item())
            s, _ = rays.evaluate_with_grad(theta)
            ss.append(s.item())
            return s
        optimiser.step(closure)
    print("Final theta:", theta)

    _, axes = plt.subplots()
    registration.generate_drr(theta.clone().detach()).display(axes)
    plt.show()

    error = fix_angle(theta - registration.true_theta)
    print("Distance: ", torch.abs(error).item())

    plt.plot(thetas)
    plt.plot(ss)
    plt.show()