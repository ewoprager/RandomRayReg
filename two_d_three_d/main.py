import sys, os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Union
import time

from two_d_three_d.data import Ray, Volume, Image
import tools
from random_rays import RandomRays, Registration


class Main:
    __init_key = object()
    def __init__(self,
                 init_key,
                 registration: Registration,
                 *,
                 source_position: torch.Tensor,
                 drr_alpha: float,
                 true_theta: Union[torch.Tensor, None]=None,
                 cache_directory: Union[str, None]=None,
                 save_or_load: Union[bool, None]=None):
        assert (init_key is self.__class__.__init_key), "Constructor is private"
        self.registration = registration
        self.source_position = source_position
        self.drr_alpha = drr_alpha
        self.true_theta = true_theta
        self.cache_directory = cache_directory
        self.save_or_load = save_or_load

        if self.save_or_load is not None and self.cache_directory is not None:
            if not self.save_or_load:
                self.registration.save_image(self.cache_directory)
                torch.save(source_position, self.cache_directory + "/source_position.pt")
                torch.save(drr_alpha, self.cache_directory + "/drr_alpha.pt")
                if true_theta is not None:
                    torch.save(self.true_theta, self.cache_directory + "/true_theta.pt")

        # display drr target
        _, axes = plt.subplots()
        self.registration.image.display(axes)
        plt.axis('square')
        plt.title("DRR at true orientation = {:.1f} (simulated X-ray intensity)".format(self.true_theta))
        plt.show()

    @classmethod
    def new_drr_registration(cls,
                             volume_path: str,
                             *,
                             device,
                             image_size: torch.Tensor,
                             source_position: torch.Tensor,
                             drr_alpha: float,
                             cache_directory: Union[str, None]=None):
        volume = Volume.from_file(volume_path, device=device)
        if cache_directory is not None:
            volume.save(cache_directory)

        registration = Registration(Ray, Image, volume, source_position=source_position)
        true_theta = registration.set_image_from_random_drr(image_size=image_size, drr_alpha=drr_alpha)
        return cls(cls.__init_key, registration, source_position=source_position, drr_alpha=drr_alpha, true_theta=true_theta, cache_directory=cache_directory, save_or_load=False)

    @classmethod
    def load(cls,
             cache_directory: str,
             *,
             device):
        volume = Volume.load(cache_directory, device=device)
        source_position = torch.load(cache_directory + "/source_position.pt")
        registration = Registration(Ray, Image, volume, source_position=source_position)
        registration.load_image(cache_directory)
        true_theta = None
        if os.path.exists(cache_directory + "/true_theta.pt"):
            true_theta = torch.load(cache_directory + "/true_theta.pt")

        return cls(cls.__init_key, registration, source_position=source_position, drr_alpha=torch.load(cache_directory + "/drr_alpha.pt"), true_theta=true_theta, cache_directory=cache_directory, save_or_load=True)

    def plot_landscape(self, *, load_rays_from_cache: bool=False):
        m: int = 3
        alpha = 0.5
        # ray_density = 500.
        # ray_count = int(np.ceil(4. * (torch.norm(self.registration.source_position) / alpha).square() * ray_density))
        ray_count = 1000000
        ray_subset_count = ray_count

        rays = None
        if load_rays_from_cache and self.cache_directory is not None:
            rays = RandomRays.load(self.registration, self.cache_directory)
            if rays.ray_count < ray_count:
                rays = None

        if rays is None:
            print("Generating {} rays...".format(ray_count))
            tic = time.time()
            rays = RandomRays.new(self.registration, integrate_alpha=self.drr_alpha, ray_count=ray_count)
            toc = time.time()
            print("Done. Took {:.3f}s".format(toc - tic))
            if self.cache_directory is not None:
                rays.save(self.cache_directory)

        blur_constant = 4.

        cmap = mpl.colormaps['viridis']

        theta_count = 200
        thetas = torch.cat((self.true_theta.value[0:5].repeat(theta_count, 1), torch.linspace(-torch.pi, torch.pi, theta_count)[:, None]), dim=1)
        landscapes = torch.zeros(m, theta_count)
        average_sum_ns = torch.zeros(m)
        print("Evaluating {} landscapes...".format(m))
        tic = time.time()
        for j in range(m):
            # alpha = .4 * 2.**j

            # ss_clipped = ss.clone()
            # ssn_clipped = 0.

            print("\tPerforming {} evaluations for alpha = {:.2f}...".format(theta_count, alpha))
            _tic = time.time()
            for i in range(theta_count):

                similarity, sum_n = rays.evaluate(Ray.Transformation(thetas[i]),
                                                  alpha=torch.tensor([alpha]),
                                                  blur_constant=torch.tensor([blur_constant]),
                                                  clip=False,
                                                  ray_count=ray_subset_count,
                                                  debug_plots=abs(thetas[i, 5] - self.true_theta.value[5]) < 0.04)
                # print(s, sn)
                landscapes[j, i] = similarity.item()
                average_sum_ns[j] += sum_n

                # s_clipped, sn_clipped = rays.evaluate(Ray.Transformation(thetas[i]),
                #                                       alpha=torch.tensor([alpha]),
                #                                       blur_constant=torch.tensor([blur_constant]),
                #                                       clip=True,
                #                                       ray_count=ray_subset_count)
                # ss_clipped[i] = s_clipped.item()
                # ssn_clipped += sn_clipped

            _toc = time.time()
            print("\tDone. Took {:.3f}s".format(_toc - _tic))

            average_sum_ns[j] /= float(theta_count)

            # asn_clipped = ssn_clipped / float(thetas.size()[0])

            alpha *= 2.

        toc = time.time()
        print("Done. Took {:.3f}s".format(toc - tic))

        _, axes = plt.subplots()

        for j in range(m):
            colour = cmap(float(j) / float(m - 1))
            axes.plot(thetas[:, 5], landscapes[j], color=colour, linestyle='-')

            # axes.plot(thetas[:, 5], ss_clipped, label="clipped, av. sum n = {:.3f}".format(asn_clipped), color=colour, linestyle='--')

        axes.vlines(self.true_theta.value[5].item(), -1., axes.get_ylim()[1])

        # plt.legend()
        plt.title("Optimisation landscape")
        plt.xlabel("theta (radians)")
        plt.ylabel("-WZNCC")
        plt.show()

    def optimise(self):
        print("True theta:", -self.true_theta.value.item())

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

        error = tools.fix_angle(theta.value + self.true_theta.value)
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
    ct_path = sys.argv[1]

    load_cached: bool = (len(sys.argv) > 2 and sys.argv[2] == "load_cached")

    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    dev = torch.device('cpu')

    if load_cached:
        main = Main.load("two_d_three_d/cache", device=dev)
    else:
        main = Main.new_drr_registration(ct_path,
                                         device=dev,
                                         image_size=torch.tensor([1000, 1000]),
                                         source_position=torch.tensor([0., 0., 11.]),
                                         drr_alpha=2000.,
                                         cache_directory="two_d_three_d/cache")

    main.plot_landscape(load_rays_from_cache=load_cached)

    exit()

    main.optimise()