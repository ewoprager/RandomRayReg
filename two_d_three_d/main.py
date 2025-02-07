import sys, os
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Union, Tuple
import time
import warnings
from tqdm import tqdm

from two_d_three_d.data import Ray, Volume, Image
import tools
from random_rays import RandomRays, Registration
import correlation_measure
import debug


class Main:
    __init_key = object()

    def __init__(self, init_key, registration: Registration, *, source_position: torch.Tensor, drr_alpha: float,
                 true_theta: Union[torch.Tensor, None] = None, cache_directory: Union[str, None] = None,
                 save_or_load: Union[bool, None] = None):
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
    def new_drr_registration(cls, volume_path: str, *, device, image_size: torch.Tensor, source_position: torch.Tensor,
                             drr_alpha: float, cache_directory: Union[str, None] = None):
        volume = Volume.from_file(volume_path, device=device)
        if cache_directory is not None:
            volume.save(cache_directory)

        registration = Registration(Ray, Image, volume, source_position=source_position)
        true_theta = registration.set_image_from_random_drr(image_size=image_size, drr_alpha=drr_alpha)
        return cls(cls.__init_key, registration, source_position=source_position, drr_alpha=drr_alpha,
            true_theta=true_theta, cache_directory=cache_directory, save_or_load=False)

    @classmethod
    def new_synthetic_drr_registration(cls, volume_size: Tuple[int, int, int], *, device, image_size: torch.Tensor,
                                       source_position: torch.Tensor, drr_alpha: float):
        volume = Volume.from_data(6000. * torch.rand(volume_size, device=device) - 1000.)
        registration = Registration(Ray, Image, volume, source_position=source_position)
        true_theta = registration.set_image_from_random_drr(image_size=image_size, drr_alpha=drr_alpha)
        return cls(cls.__init_key, registration, source_position=source_position, drr_alpha=drr_alpha,
            true_theta=true_theta)

    @classmethod
    def load(cls, cache_directory: str, *, device, regenerate_drr: bool = False,
             image_size: Union[torch.Tensor, None] = None):
        volume = Volume.load(cache_directory, device=device)
        source_position = torch.load(cache_directory + "/source_position.pt")
        registration = Registration(Ray, Image, volume, source_position=source_position)
        drr_alpha = torch.load(cache_directory + "/drr_alpha.pt")
        # loading ground truth transformation if it exists
        true_theta = None
        if os.path.exists(cache_directory + "/true_theta.pt"):
            true_theta = torch.load(cache_directory + "/true_theta.pt")
        # regenerating the DRR if requested and possible
        if regenerate_drr:
            if true_theta is None:
                print("Cannot regenerate DRR with existing transformation, as none exists")
                regenerate_drr = False
            if image_size is None:
                print("To regenerate the DRR, an `image_size` must be provided")
                regenerate_drr = False
        if regenerate_drr:
            registration.set_image_from_drr(true_theta, image_size=image_size, drr_alpha=drr_alpha)
            registration.save_image(cache_directory)
        else:
            registration.load_image(cache_directory)

        return cls(cls.__init_key, registration, source_position=source_position, drr_alpha=drr_alpha,
            true_theta=true_theta, cache_directory=cache_directory, save_or_load=True)

    def __get_rays(self, ray_count, *, load_rays_from_cache: bool = False):
        if load_rays_from_cache and self.cache_directory is not None:
            ret = RandomRays.load(self.registration, self.cache_directory)
            if ret.ray_count >= ray_count:
                return ret

        debug.tic("Generating {} rays".format(ray_count))
        ret = RandomRays.new(self.registration, integrate_alpha=self.drr_alpha, ray_count=ray_count)
        debug.toc()
        if self.cache_directory is not None:
            ret.save(self.cache_directory)
        return ret

    def check_match(self, *, load_rays_from_cache: bool = False):
        assert (self.true_theta is not None), "Cannot check match with no ground truth alignment"

        alpha = .5
        blur_constant = 4.
        ray_count = 15000000

        rays = self.__get_rays(ray_count, load_rays_from_cache=load_rays_from_cache)

        similarity, weight_sum = rays.evaluate(self.true_theta, alpha=torch.tensor([alpha]),
            blur_constant=torch.tensor([blur_constant]), clip=False, ray_count=ray_count, debug_plots=True)

        print("Similarity: {:.3f}", similarity)

        print("Sum n = {:.3f}", weight_sum)

    def plot_landscape(self, *, load_rays_from_cache: bool = True):
        assert (self.true_theta is not None), "Cannot plot landscape with no ground truth alignment"

        m: int = 4
        # alpha = 3.
        expected_weight_sum = 5000.
        # ray_count = int(np.ceil(4. * (torch.norm(self.registration.source_position) / alpha).square() * expected_weight_sum))
        ray_count = 1000000
        ray_subset_count = ray_count

        rays = self.__get_rays(ray_count, load_rays_from_cache=load_rays_from_cache)

        blur_constant = 1.

        cmap = mpl.colormaps['viridis']

        theta_count = 50
        thetas = torch.cat((
        self.true_theta.value[0:5].repeat(theta_count, 1), torch.linspace(-torch.pi, torch.pi, theta_count)[:, None]),
            dim=1)

        # for analysing new method
        landscapes = torch.zeros(m, theta_count)
        average_weight_sums = torch.zeros(m)
        # landscapes_clipped = landscapes.clone()
        # average_weight_sums_clipped = average_weight_sums.clone()

        # for analysing canonical method
        landscapes_canonical = torch.zeros(m, theta_count)
        canonical_mip_level = 0

        # for timing both
        times = torch.zeros(m)
        times_canonical = torch.zeros(m)

        debug.tic("Evaluating {} landscapes".format(m))
        for j in range(m):

            # ss_clipped = ss.clone()
            # ssn_clipped = 0.

            alpha = torch.norm(self.registration.source_position) * torch.sqrt(
                torch.tensor([2. * expected_weight_sum / float(ray_subset_count)]))

            debug.tic("Performing {} evaluations for {} rays".format(theta_count, ray_subset_count))
            for i in tqdm(range(theta_count), desc=debug.get_indent()):
                tic = time.time()
                similarity, weight_sum = rays.evaluate(Ray.Transformation(thetas[i]), alpha=torch.tensor([alpha]),
                    blur_constant=torch.tensor([blur_constant]), ray_count=ray_subset_count)
                times[j] += time.time() - tic
                # print(s, sn)
                landscapes[j, i] = similarity.item()
                average_weight_sums[j] += weight_sum

                # similarity_clipped, sum_n_clipped = rays.evaluate(Ray.Transformation(thetas[i]),
                #                                                   alpha=torch.tensor([alpha]),
                #                                                   blur_constant=torch.tensor([blur_constant]),
                #                                                   clip=True,
                #                                                   ray_count=ray_subset_count)
                # landscapes_clipped[j, i] = similarity_clipped.item()
                # average_weight_sums_clipped += sum_n_clipped

                tic = time.time()
                landscapes_canonical[j, i] = self.canonical(Ray.Transformation(thetas[i]),
                    self.registration.image.data[canonical_mip_level], volume_mip_level=canonical_mip_level)
                times_canonical[j] += time.time() - tic

            debug.toc()

            average_weight_sums[j] /= float(theta_count)
            # average_weight_sums_clipped[j] /= float(theta_count)

            # alpha *= 2.
            ray_subset_count = ray_subset_count // 16
            canonical_mip_level += 2

        debug.toc()

        _, axes = plt.subplots()

        for j in range(m):
            colour = cmap(float(j) / float(m - 1) if m > 1 else 0.5)
            axes.plot(thetas[:, 5], landscapes[j], color=colour, linestyle='-',
                label="{}; {:.3f}s".format(j, times[j].item()))
            # axes.plot(thetas[:, 5], landscapes_clipped[j], label="clipped, av. sum n = {:.3f}".format(average_weight_sums_clipped[j]), color=colour, linestyle='--')
            axes.plot(thetas[:, 5], landscapes_canonical[j], color=colour, linestyle='--',
                label="canonical {}; {:.3f}s".format(j, times_canonical[j].item()))

        axes.vlines(self.true_theta.value[5].item(), -1., axes.get_ylim()[1])

        plt.legend()
        plt.title("Optimisation landscape")
        plt.xlabel("theta (radians)")
        plt.ylabel("-WZNCC")
        plt.show()

    def plot_distance_correlation(self, point_count: int, *, load_rays_from_cache: bool = True):
        assert (self.true_theta is not None), "Cannot plot distance correlation with no ground truth alignment"

        m = 5

        alpha = .2

        ray_count = 1000000
        ray_subset_count = ray_count

        rays = self.__get_rays(ray_count, load_rays_from_cache=load_rays_from_cache)

        blur_constant = 4.

        print("Plotting for {} values of alpha...".format(m))
        tic = time.time()
        for j in range(m):
            distances = torch.zeros(point_count)
            nwznccs = torch.zeros(point_count)
            theta = Ray.Transformation()
            print("\tEvaluating {} random points for alpha = {:.3f}...".format(point_count, alpha))
            _tic = time.time()
            for i in range(point_count):
                theta.randomise()
                similarity, _ = rays.evaluate(theta, alpha=torch.tensor([alpha]),
                    blur_constant=torch.tensor([blur_constant]), ray_count=ray_subset_count)
                distances[i] = self.true_theta.distance(theta)
                nwznccs[i] = similarity.item()

            _toc = time.time()
            print("\tDone; took {:.3f}s.".format(_toc - _tic))

            cc = torch.corrcoef(torch.cat((distances[None, :], nwznccs[None, :])))

            plt.scatter(distances, nwznccs, s=.3)
            plt.xlabel("Riemann distance")
            plt.ylabel("-WZNCC")
            plt.title("alpha = {:.3f}; correlation = {}".format(alpha, cc[0, 1]))
            plt.show()

            print("\tCorrelation coefficient = {}".format(cc))

            alpha *= 1.5
        toc = time.time()
        print("Done; took {:.3f}s.".format(toc - tic))

    def quantify_distance_correlation(self, load_rays_from_cache: bool = True):
        assert (self.true_theta is not None), "Cannot quantify distance correlation with no ground truth alignment"

        def se3_from_distance(distance: float) -> Ray.Transformation:
            angle = distance * torch.rand(1)
            r = angle * torch.nn.functional.normalize(torch.rand(1, 3) - .5)
            norm_t = torch.sqrt(distance * distance - angle * angle)
            t = norm_t * torch.nn.functional.normalize(torch.rand(1, 3) - .5)
            return Ray.Transformation(torch.cat((t, r), dim=-1)[0])

        alpha = 1.
        blur_constant = 4.
        ray_count = 100000
        ray_subset_count = ray_count

        rays = self.__get_rays(ray_count, load_rays_from_cache=load_rays_from_cache)

        debug.tic("Quantifying correlation for alpha = {:.3f}".format(alpha))

        def similarity_from_distance(distance: float) -> float:
            theta = self.true_theta.compose(se3_from_distance(distance))
            similarity, _ = rays.evaluate(theta, alpha=torch.tensor([alpha]),
                blur_constant=torch.tensor([blur_constant]), ray_count=ray_subset_count)
            return similarity.item()

        cc = correlation_measure.quantify_correlation(similarity_from_distance, (0., 2.))

        debug.toc("Correlation coefficient = {:.3f}".format(cc))

    def optimise(self):
        print("True theta:", -self.true_theta.value)

        alpha = 2.
        # expected_weight_sum = 100.
        blur_constant = 4.

        # ray_count = int(np.ceil(torch.norm(self.registration.source_position) * expected_weight_sum * np.sqrt(np.pi) / alpha))

        ray_count = 1000000

        rays = self.__get_rays(ray_count, load_rays_from_cache=True)

        thetas = []
        similarities = []
        theta = Ray.Transformation()
        theta.randomise()

        _, axes = plt.subplots()
        Image(self.registration.generate_drr(theta, alpha=self.drr_alpha, image_size=torch.tensor([300, 300]))).display(
            axes)
        plt.axis('square')
        plt.title("DRR at initial orientation = {} (simulated X-ray intensity)".format(theta.value))
        plt.show()

        optimiser = torch.optim.SGD([theta.value], lr=0.005, momentum=0.9, differentiable=True)
        print("Optimising...")
        tic = time.time()
        for i in range(350):
            def closure():
                optimiser.zero_grad()
                thetas.append(theta.value)
                similarity, _ = rays.evaluate_with_grad(theta, alpha=torch.tensor([alpha]),
                    blur_constant=torch.tensor([blur_constant]))
                similarities.append(similarity.item())
                print("\tStep {}; similarity = {:.3f} at theta distance = {:.3f}".format(i, similarity.item(),
                    self.true_theta.distance(theta).item()))
                return similarity

            optimiser.step(closure)
        toc = time.time()
        print("Done; took {:.3f}s.".format(toc - tic))
        print("Final theta:", theta.value)

        _, axes = plt.subplots()
        Image(self.registration.generate_drr(theta, alpha=self.drr_alpha, image_size=torch.tensor([300, 300]))).display(
            axes)
        plt.axis('square')
        plt.title("DRR at final orientation = {} (simulated X-ray intensity)".format(theta.value))
        plt.show()

        # error = theta.value + self.true_theta.value
        # print("Distance: ", torch.abs(error).item())

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Optimisation step")

        colour = 'green'
        ax1.plot(similarities, color=colour)
        ax1.set_ylabel("-WZNCC", color=colour)
        ax1.tick_params(axis='y', labelcolor=colour)

        # colour = 'blue'
        # ax2 = ax1.twinx()
        # ax2.plot(thetas, label="orientation", color=colour)
        # ax2.set_ylabel("orientation (radians)", color=colour)
        # ax2.tick_params(axis='y', labelcolor=colour)

        plt.title("Optimisation process")
        plt.legend()
        fig.tight_layout()
        plt.show()

    def canonical(self, theta, fixed_image_data: torch.Tensor, *, volume_mip_level: int = 0) -> float:
        assert (self.true_theta is not None), "Cannot assess similarity canonically with no ground truth alignment"

        drr = Image(self.registration.generate_drr(theta, alpha=self.drr_alpha,
            image_size=torch.tensor(fixed_image_data.size()), mip_level=volume_mip_level))

        xs = fixed_image_data.flatten()
        ys = drr.data[0].flatten()
        return -tools.weighted_zero_normalised_cross_correlation(xs, ys, torch.ones_like(xs)).item()

    def quantify_distance_correlation_canonical(self) -> float:
        assert (
          self.true_theta is not None), "Cannot quantify canonical distance correlation with no ground truth alignment"

        def se3_from_distance(distance: float) -> Ray.Transformation:
            angle = distance * torch.rand(1)
            r = angle * torch.nn.functional.normalize(torch.rand(1, 3) - .5)
            norm_t = torch.sqrt(distance * distance - angle * angle)
            t = norm_t * torch.nn.functional.normalize(torch.rand(1, 3) - .5)
            return Ray.Transformation(torch.cat((t, r), dim=-1)[0])

        debug.tic("Quantifying correlation for canonical method")

        def similarity_from_distance(distance: float) -> float:
            theta = self.true_theta.compose(se3_from_distance(distance))
            similarity = self.canonical(theta, self.registration.image.data[3], volume_mip_level=3)
            return similarity

        cc = correlation_measure.quantify_correlation(similarity_from_distance, (0., 2.))

        debug.toc("Correlation coefficient = {:.3f}".format(cc))

    def fourier_grangeat(self, drr_alpha: float):
        """
        ! Note: this assumes that the source position is on the z-axis !
        :return:
        """
        assert (self.true_theta is not None), "Cannot do Fourier / Grangeat analysis with no ground truth alignment"

        mip_level = 0

        size = self.registration.image.data[mip_level].size()
        xs = 2. * torch.arange(0, size[1], 1, dtype=torch.float32) / float(size[1] - 1) - 1.
        ys = 2. * torch.arange(0, size[0], 1, dtype=torch.float32) / float(size[0] - 1) - 1.
        ys, xs = torch.meshgrid(ys, xs)
        sq_mags = xs * xs + ys * ys
        norm = torch.sqrt(sq_mags + 1e-8)
        xs_normalised = xs / norm
        ys_normalised = ys / norm

        source_dist: float = self.registration.source_position[2].item()

        g = -drr_alpha * torch.log(1. - self.registration.image.data[mip_level])

        g_tilde = g * source_dist / torch.sqrt(source_dist * source_dist + sq_mags)

        dy, dx = torch.gradient(g_tilde)
        radial_derivative = dx * xs_normalised + dy * ys_normalised

        fixed_scaling: torch.Tensor = sq_mags / (source_dist * source_dist) + 1.
        rhs = torch.fft.fft2(radial_derivative * fixed_scaling)

        plt.pcolormesh(rhs.abs().log(), cmap='gray')
        plt.axis('square')
        plt.show()

        size_ct = self.registration.volume.data[mip_level].size()
        xs_ct = 2. * torch.arange(0, size_ct[2], 1, dtype=torch.float32) / float(size_ct[2] - 1) - 1.
        ys_ct = 2. * torch.arange(0, size_ct[1], 1, dtype=torch.float32) / float(size_ct[1] - 1) - 1.
        zs_ct = 2. * torch.arange(0, size_ct[0], 1, dtype=torch.float32) / float(size_ct[0] - 1) - 1.
        zs_ct, ys_ct, xs_ct = torch.meshgrid(zs_ct, ys_ct, xs_ct)
        sq_mags_ct = xs_ct * xs_ct + ys_ct * ys_ct + zs_ct * zs_ct
        norm_ct = torch.sqrt(sq_mags_ct + 1e-8)
        xs_normalised_ct = xs_ct / norm_ct
        ys_normalised_ct = ys_ct / norm_ct
        zs_normalised_ct = zs_ct / norm_ct

        # omega_x = torch.fft.fftfreq(size_ct[2])
        # omega_y = torch.fft.fftfreq(size_ct[1])
        # omega_z = torch.fft.fftfreq(size_ct[0])
        # omega_z, omega_y, omega_x = torch.meshgrid(omega_z, omega_y, omega_x)
        # omega_radial = omega_x * xs_normalised_ct + omega_y * ys_normalised_ct + omega_z * zs_normalised_ct
        # print(omega_radial)
        # vol = omega_radial * 1j * torch.fft.fftn(self.registration.volume.data[mip_level])

        dz_ct, dy_ct, dx_ct = torch.gradient(self.registration.volume.data[mip_level])
        radial_derivative_ct = dx_ct * xs_normalised_ct + dy_ct * ys_normalised_ct + dz_ct * zs_normalised_ct
        vol = torch.fft.fftn(radial_derivative_ct)

        zs = torch.zeros_like(xs)
        ws = torch.ones_like(xs)
        positions = torch.stack([xs, ys, zs, ws], dim=-1)
        positions_reshaped = positions.view(-1, 4)

        def evaluate(theta, plot: bool = False) -> float:
            positions_transformed = torch.matmul(positions_reshaped, theta.inverse().get_matrix().T)
            positions_transformed = positions_transformed.view(positions.shape)[:, :, 0:3]
            dirs = torch.nn.functional.normalize(positions_transformed - self.registration.source_position, dim=-1)
            lambdas = -torch.inner(dirs, self.registration.source_position)
            grid = self.registration.source_position + lambdas[:, :, None] * dirs

            lhs_real = torch.nn.functional.grid_sample(vol.real[None, None, :, :, :], grid[None, None, :, :, :])[
                0, 0, 0]
            lhs_imag = torch.nn.functional.grid_sample(vol.imag[None, None, :, :, :], grid[None, None, :, :, :])[
                0, 0, 0]
            lhs = lhs_real + 1j * lhs_imag

            if plot:
                plt.pcolormesh(lhs.abs().log(), cmap='gray')
                plt.axis('square')
                plt.show()

            l = lhs.real.flatten()
            return -tools.weighted_zero_normalised_cross_correlation(l, rhs.real.flatten(), torch.ones_like(l)).item()

        print("At ground truth: {:.3e}".format(evaluate(self.true_theta, plot=True)))

        cmap = mpl.colormaps['viridis']

        m = 1

        theta_count = 200
        thetas = torch.cat((self.true_theta.value[0:5].repeat(theta_count, 1),
        torch.linspace(-torch.pi, torch.pi, theta_count)[:, None]), dim=1)

        landscapes = torch.zeros(m, theta_count)

        # for timing
        times = torch.zeros(m)

        debug.tic("Evaluating {} landscapes".format(m))
        for j in range(m):
            debug.tic("Performing {} evaluations".format(theta_count))
            for i in tqdm(range(theta_count), desc=debug.get_indent()):
                tic = time.time()
                similarity = evaluate(Ray.Transformation(thetas[i]))
                times[j] += time.time() - tic
                landscapes[j, i] = similarity

            debug.toc()  # mip_level += 2

        debug.toc()

        _, axes = plt.subplots()

        for j in range(m):
            colour = cmap(float(j) / float(m - 1) if m > 1 else 0.5)
            axes.plot(thetas[:, 5], landscapes[j], color=colour, linestyle='-',
                label="{}; {:.3f}s".format(j, times[j].item()))

        axes.vlines(self.true_theta.value[5].item(), -1., axes.get_ylim()[1])

        plt.legend()
        plt.title("Optimisation landscape")
        plt.xlabel("theta (radians)")
        plt.ylabel("-WZNCC")
        plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ct_path = sys.argv[1]
    debug.init()

    load_cached: bool = (len(sys.argv) > 2 and sys.argv[2] == "load_cached")

    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    dev = torch.device('cpu')
    drr_size = torch.tensor([1000, 1000])

    if load_cached:
        main = Main.load("two_d_three_d/cache", device=dev)
    else:
        main = Main.new_drr_registration(ct_path, device=dev, image_size=drr_size,
            source_position=torch.tensor([0., 0., 11.]), drr_alpha=2000., cache_directory="two_d_three_d/cache")

    # main = Main.new_synthetic_drr_registration((6, 6, 8), device=dev, image_size=torch.tensor([12, 12]),
    #     source_position=torch.tensor([0., 0., 11.]), drr_alpha=2000.)

    main.fourier_grangeat(drr_alpha=2000.)

    # main.plot_landscape(load_rays_from_cache=load_cached)

    # main.optimise()

    # main.plot_distance_correlation(10000)

    # main.quantify_distance_correlation()  # main.quantify_distance_correlation_canonical()
