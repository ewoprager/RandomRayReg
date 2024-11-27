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
    image_size: int = 10
    volume_size: int = 6

    # CT space is 2D, exists between (-1,-1) and (1,1)
    volume_data = torch.rand(volume_size, volume_size)
    volume = data.Volume(volume_data)

    registration = Registration(volume, image_size, source_position=torch.tensor([3., 0.]))

    # display drr target
    _, ax0 = plt.subplots()
    registration.image.display(ax0)
    plt.show()

    # display volume
    _, ax1 = plt.subplots()
    registration.volume.display(ax1)
    ax1.set_ylim(-1.5, 1.5)
    plt.show()

    # rays = RandomRays(registration)

    # display volume, with rays colours according to fixed image brightness
    # _, ax1 = plt.subplots()
    # registration.volume.display(ax1)
    # rays.plot_with_sampled_shading(ax1)
    # ax1.set_ylim(-1.5, 1.5)
    # plt.show()

    # display volume, with rays coloured according to volume integral
    # _, ax2 = plt.subplots()
    # registration.volume.display(ax2)
    # rays.plot_with_intensity_shading(ax2)
    # ax2.set_ylim(-1.5, 1.5)
    # plt.show()

    ray_count = 100
    while ray_count < 100000:
        rays = RandomRays(registration, ray_count=ray_count)
        thetas = np.linspace(-torch.pi, torch.pi, 200, dtype=np.float32)
        ss = np.zeros_like(thetas)
        ssn = 0.
        for i in range(len(thetas)):
            s, sn = rays.evaluate(torch.tensor([thetas[i]]), alpha=torch.tensor([500.]))
            ss[i] = s.item()
            ssn += sn
        asn = ssn / float(len(thetas))
        plt.plot(thetas, ss)
        plt.vlines(registration.true_theta.item(), -1., 1.)
        plt.title("{} rays, av. sum n = {:.3f}".format(ray_count, asn))
        plt.show()
        ray_count *= 2

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