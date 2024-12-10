import torch
import time
import matplotlib.pyplot as plt
from typing import Tuple, Callable


def quantify_correlation(function: Callable[[float], float],
                         domain: Tuple[float, float],
                         *,
                         divisions: int=20,
                         sample_count: int=50) -> float:
    inputs = torch.linspace(domain[0], domain[1], divisions)
    samples = torch.zeros(divisions, sample_count)

    print("Taking {} samples ({} samples in each of {} divisions of the domain)...".format(divisions * sample_count, sample_count, divisions))
    tic = time.time()
    for d in range(divisions):
        for i in range(sample_count):
            samples[d, i] = function(inputs[d].item())
    toc = time.time()
    print("Done; took {:.3f}s".format(toc - tic))

    print("Evaluating means and standard deviations...")
    tic = time.time()
    stds, means = torch.std_mean(samples, dim=-1)
    toc = time.time()
    print("Done; took {:.3f}s".format(toc - tic))

    print("Evaluating correlation coefficient...")
    tic = time.time()
    xs = inputs[:, None].repeat(1, sample_count).reshape(1, divisions * sample_count)
    ys = samples.reshape(1, divisions * sample_count)
    cc = torch.corrcoef(torch.cat((xs, ys)))
    toc = time.time()
    print("Done; took {:.3f}s".format(toc - tic))

    ret = cc[0, 1].item()

    plt.scatter(inputs, means)
    plt.errorbar(inputs, means, yerr=stds)
    plt.xlabel("Riemann distance")
    plt.ylabel("-WZNCC")
    plt.title("Estimated similarity against Riemann distance with even sampling in the domain; CC = {:.3f}".format(ret))
    plt.show()

    return ret