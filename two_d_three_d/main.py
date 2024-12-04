import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import nrrd
import gc

from two_d_three_d.data import Ray, Volume, Image
import tools
from random_rays import RandomRays, Registration


class Main:
    def __init__(self,
                 image_size: torch.Tensor,
                 volume_data: torch.Tensor,
                 drr_alpha: float=.5):
        self.image_size = image_size
        self.volume_data = volume_data
        # self.volume_size = volume_size

        # CT space is 2D, exists between (-1,-1) and (1,1)
        self.volume = Volume(self.volume_data)

        self.registration = Registration(Ray, Image, self.volume, image_size, source_position=torch.tensor([0., 0., 11.]), drr_alpha=drr_alpha)

        # display drr target
        _, ax0 = plt.subplots()
        self.registration.image.display(ax0)
        plt.title("DRR at true orientation = {:.3f} (simulated X-ray intensity)".format(self.registration.true_theta))
        plt.show()

        # display volume
        # _, ax1 = plt.subplots()
        # self.registration.volume.display(ax1)
        # ax1.set_ylim(-1., 1.)
        # plt.title("CT volume (X-ray attenuation coefficient)")
        # plt.show()

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
        m: int = 5
        _, axes = plt.subplots()
        alpha = 1.
        # ray_density = 500.
        # ray_count = int(np.ceil(4. * (torch.norm(self.registration.source_position) / alpha).square() * ray_density))
        max_ray_count = 3000
        rays = RandomRays(self.registration, ray_count=max_ray_count)
        rays.save("two_d_three_d/cache")

        ray_count = max_ray_count

        blur_constant = 4.

        cmap = mpl.colormaps['viridis']

        theta_count = 200
        thetas = torch.cat((self.registration.true_theta.value[0:5].repeat(theta_count, 1), torch.linspace(-torch.pi, torch.pi, theta_count)[:, None]), dim=1)
        for j in range(m):
            # alpha = .4 * 2.**j

            ss = torch.zeros(theta_count)
            ssn = 0.
            # ss_clipped = ss.clone()
            # ssn_clipped = 0.
            for i in range(theta_count):
                s, sn = rays.evaluate(Ray.Transformation(thetas[i]),
                                      alpha=torch.tensor([alpha]),
                                      blur_constant=torch.tensor([blur_constant]),
                                      clip=False,
                                      ray_count=ray_count)
                # print(s, sn)
                ss[i] = s.item()
                ssn += sn

                # s_clipped, sn_clipped = rays.evaluate(Ray.Transformation(thetas[i]),
                #                                       alpha=torch.tensor([alpha]),
                #                                       blur_constant=torch.tensor([blur_constant]),
                #                                       clip=True)
                # ss_clipped[i] = s_clipped.item()
                # ssn_clipped += sn_clipped

                gc.collect()

                ray_count = (2 * ray_count) // 3

            asn = ssn / float(theta_count)
            # asn_clipped = ssn_clipped / float(thetas.size()[0])
            colour = cmap(float(j) / float(m - 1))
            axes.plot(thetas[:, 5], ss,
                      label="alpha = {:.3f}, {} rays, av. sum n = {:.3f}, bc = {:.3f}".format(alpha, ray_count, asn,
                                                                                              blur_constant),
                      color=colour, linestyle='-')

            # axes.plot(thetas, ss_clipped, label="clipped, av. sum n = {:.3f}".format(asn_clipped), color=colour, linestyle='--')

        axes.vlines(-self.registration.true_theta.value[5].item(), -1., axes.get_ylim()[1])
        # ray_density *= 2.

        # plt.legend()
        plt.title("Optimisation landscape")
        plt.xlabel("theta (radians)")
        plt.ylabel("-WZNCC")
        plt.show()

    def optimise(self):
        print("True theta:", -self.registration.true_theta.value.item())

        alpha = 1.
        ray_density = 100.
        blur_constant = 4.

        ray_count = int(np.ceil(torch.norm(self.registration.source_position) * ray_density * np.sqrt(np.pi) / alpha))
        rays = RandomRays(self.registration, ray_count=ray_count)

        thetas = []
        ss = []
        theta = Ray.Transformation()
        theta.randomise()
        optimiser = torch.optim.SGD([theta.value], lr=1.5, momentum=0.75)
        for i in range(50):
            def closure():
                optimiser.zero_grad()
                thetas.append(tools.fix_angle(theta.value).item())
                s, _ = rays.evaluate_with_grad(theta,
                                               alpha=torch.tensor([alpha]),
                                               blur_constant=torch.tensor([blur_constant]))
                ss.append(s.item())
                return s

            optimiser.step(closure)
        print("Final theta:", tools.fix_angle(theta.value))

        _, axes = plt.subplots()
        self.registration.generate_drr(theta).display(axes)
        plt.title("DRR at final orientation = {:.3f} (simulated X-ray intensity)".format(theta.value.item()))
        plt.show()

        error = tools.fix_angle(theta.value + self.registration.true_theta.value)
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
    sys.settrace(None)

    if len(sys.argv) != 2:
        print("Please pass a single argument: a path to a CT volume file.")
        exit(1)

    ct_path = sys.argv[1]

    data, header = nrrd.read(ct_path)

    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cpu')

    volume_data = torch.maximum(torch.tensor(data, device=device).type(torch.float32) + 1000., torch.tensor([0.], device=device))

    main = Main(image_size=torch.tensor([100, 100]), volume_data=volume_data, drr_alpha=2000.)

    main.plot_landscape()

    exit()

    main.optimise()