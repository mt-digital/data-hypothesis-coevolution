##
# Run multiple coevolution_simulations with d, h, and W of varying
# cultural complexities, K, and varying the "peakiness" in W by varying the
# standard deviation of noise used to generate W.
#
import numpy as np

from model import coevolution_simulation

class Simulation:

    def __init__(self, K, sigma):

        d = np.abs(np.random.normal(size=(K,)))
        self.d = d / d.sum()

        h = np.abs(np.random.normal(size=(K,)))
        self.h = h / h.sum()

        self.W = np.abs(np.random.normal(scale=sigma, size=(K, K)) + 0.0001)

        self.d_hist = None
        self.h_hist = None

    def run(self, conv_limit=1e-4, max_its=1e4):

        self.d_hist, self.h_hist = \
            coevolution_simulation(self.d, self.h, self.W)

        return (self.d_hist, self.h_hist)


class Experiment:

    def __init__(self, Ks, sigmas, n_trials=10):

        self.Ks = Ks
        self.sigmas = sigmas
        self.n_trials = n_trials

    def run(self, conv_limit=1e-4, max_its=1e4):

        self.trials = {}

        for K in self.Ks:
            for sigma in self.sigmas:
                for idx in range(self.n_trials):
                    # print('K={}, sigma={}, trial_idx={}'.format(K, sigma, idx))

                    sim = Simulation(K, sigma)

                    res = sim.run(conv_limit, max_its)
                    res_dict = {
                        'd': res[0],
                        'h': res[1]
                    }

                    if idx == 0:
                        self.trials[(K, sigma)] = [res_dict]
                    else:
                        self.trials[(K, sigma)].append(res_dict)
