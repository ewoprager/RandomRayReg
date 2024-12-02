import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

import data
import tools
from tools import fix_angle
from registration import Registration
from random_rays import RandomRays


class Main:
    def __init__(self, image_size: int, volume_size: int):
        self.image_size = image_size
        self.volume_size = volume_size

        # CT space is 2D, exists between (-1,-1) and (1,1)
        self.volume_data = torch.rand(volume_size, volume_size)
        radius = 0.5 * float(volume_size - 1)
        for i in range(self.volume_size):
            for j in range(self.volume_size):
                di = float(i) - radius
                dj = float(j) - radius
                if di * di + dj * dj > radius * radius:
                    self.volume_data[i, j] = 0.
        self.volume = data.Volume(self.volume_data)

        self.registration = Registration(self.volume, image_size, source_position=torch.tensor([3., 0.]))

        # display drr target
        _, ax0 = plt.subplots()
        self.registration.image.display(ax0)
        plt.title("DRR at true orientation = {:.3f} (simulated X-ray intensity)".format(-self.registration.true_theta.item()))
        plt.show()

        # display volume
        _, ax1 = plt.subplots()
        self.registration.volume.display(ax1)
        ax1.set_ylim(-1., 1.)
        plt.title("CT volume (X-ray attenuation coefficient)")
        plt.show()

        # display volume, with rays colours according to fixed image brightness
        # _, ax1 = plt.subplots()
        # self.registration.volume.display(ax1)
        # rays.plot_with_sampled_shading(ax1)
        # ax1.set_ylim(-1., 1.)
        # plt.show()

        # display volume, with rays coloured according to volume integral
        # _, ax2 = plt.subplots()
        # self.registration.volume.display(ax2)
        # rays.plot_with_intensity_shading(ax2)
        # ax2.set_ylim(-1., 1.)
        # plt.show()

    def plot_landscape(self):
        m: int = 3
        _, axes = plt.subplots()
        alpha = 1.
        ray_density = 1000.
        ray_count = int(np.ceil(torch.norm(self.registration.source_position) * ray_density * np.sqrt(np.pi) / alpha))
        rays = RandomRays(self.registration, ray_count=ray_count)
        blur_constant = 4.
        # for j in range(m):
        cmap = mpl.colormaps['viridis']

        thetas = np.linspace(-torch.pi, torch.pi, 200, dtype=np.float32)
        for j in range(m):
            alpha = 0.5 * 1.75**j

            ss = np.zeros_like(thetas)
            ssn = 0.
            ss_clipped = np.zeros_like(thetas)
            ssn_clipped = 0.
            for i in range(len(thetas)):
                s, sn = rays.evaluate(torch.tensor([thetas[i]]),
                                      alpha=torch.tensor([alpha]),
                                      blur_constant=torch.tensor([blur_constant]),
                                      clip=False)
                ss[i] = s.item()
                ssn += sn

                s_clipped, sn_clipped = rays.evaluate(torch.tensor([thetas[i]]),
                                                      alpha=torch.tensor([alpha]),
                                                      blur_constant=torch.tensor([blur_constant]),
                                                      clip=True)
                ss_clipped[i] = s_clipped.item()
                ssn_clipped += sn_clipped

            asn = ssn / float(len(thetas))
            asn_clipped = ssn_clipped / float(len(thetas))
            colour = cmap(float(j) / float(m - 1))
            axes.plot(thetas, ss,
                      label="alpha = {:.3f}, {} rays, av. sum n = {:.3f}, bc = {:.3f}".format(alpha, ray_count, asn,
                                                                                              blur_constant),
                      color=colour, linestyle='-')

            axes.plot(thetas, ss_clipped, label="clipped, av. sum n = {:.3f}".format(asn_clipped), color=colour, linestyle='--')

        axes.vlines(-self.registration.true_theta.item(), -1., axes.get_ylim()[1])
        # ray_density *= 2.

        # plt.legend()
        plt.title("Optimisation landscape")
        plt.xlabel("theta (radians)")
        plt.ylabel("-WZNCC")
        plt.show()

    def optimise(self):
        print("True theta:", -self.registration.true_theta)

        alpha = 1.
        ray_density = 1000.
        blur_constant = 4.

        ray_count = int(np.ceil(torch.norm(self.registration.source_position) * ray_density * np.sqrt(np.pi) / alpha))
        rays = RandomRays(self.registration, ray_count=ray_count)

        thetas = []
        ss = []
        theta = torch.pi * (-1. + 2. * torch.rand(1))
        optimiser = torch.optim.SGD([theta], lr=1.5, momentum=0.75)
        for i in range(50):
            def closure():
                optimiser.zero_grad()
                thetas.append(fix_angle(theta).item())
                s, _ = rays.evaluate_with_grad(theta,
                                               alpha=torch.tensor([alpha]),
                                               blur_constant=torch.tensor([blur_constant]))
                ss.append(s.item())
                return s

            optimiser.step(closure)
        print("Final theta:", tools.fix_angle(theta))

        _, axes = plt.subplots()
        self.registration.generate_drr(theta.clone().detach()).display(axes)
        plt.title("DRR at final orientation = {:.3f} (simulated X-ray intensity)".format(theta.item()))
        plt.show()

        error = fix_angle(theta + self.registration.true_theta)
        print("Distance: ", torch.abs(error).item())

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Optimisation step")

        colour = 'blue'
        ax1.plot(thetas, label="orientation", color=colour)
        ax1.set_ylabel("orientation (radians)", color=colour)
        ax1.tick_params(axis='y', labelcolor=colour)

        ax2 = ax1.twinx()
        colour = 'green'
        ax2.plot(ss, color=colour)
        ax2.set_ylabel("-WZNCC", color=colour)
        ax2.tick_params(axis='y', labelcolor=colour)

        plt.title("Optimisation process")
        plt.legend()
        fig.tight_layout()
        plt.show()



if __name__ == "__main__":
    main = Main(image_size=16, volume_size=8)

    main.plot_landscape()

    main.optimise()