##
# Model components for Ferdinand (2018) implementation
# (https://www.researchgate.net/publication/324479016_The_coevolution_of_data_and_hypotheses_in_bayesian_cultural_evolution)
#
import numpy as np


def iterate_d(d, h, W):

    Z_d = W.T @ d

    divisors = np.array([Z_d for _ in range(len(d))])
    D = W / divisors

    return (D @ h) * d


def iterate_h(d, h, W):

    Z_h = W @ h

    divisors = np.array([Z_h for _ in range(len(d))]).T
    H = W / divisors

    return (H.T @ d) * h


def iterate_dh(d, h, W):

    d_next = iterate_d(d, h, W)
    h_next = iterate_h(d_next, h, W)

    return (d_next, h_next)


def coevolution_simulation(d_init, h_init, W, conv_limit=1e-4, max_its=1e4):

    d_hist = np.array([d_init])
    h_hist = np.array([h_init])
    d = d_init
    h = h_init

    diff = 1.0
    its = 0
    while diff > conv_limit or its > max_its:

        d_next, h_next = iterate_dh(d, h, W)

        diff = sum(
            np.linalg.norm(x[1] - x[0])
            for x in [(d_next, d), (h_next, h)]
        )

        d = d_next
        h = h_next

        d_hist = np.append(d_hist, d)
        h_hist = np.append(h_hist, h)

        its += 1

    n = its + 1
    K = len(d)

    if its > max_its:
        print('Warning: max its exceeded for K=', str(K))
        return None

    return (d_hist.reshape(n, K), h_hist.reshape(n, K))


def test_iterate_dh():
    '''
    d and h must always sum to 1.0 at all times during iteration
    '''

    d = [0.1, 0.2, 0.3, 0.4]
    h = [0.4, 0.1, 0.25, 0.25]

    # Add small bias in order to avoid near-zeros.
    W = np.abs(np.random.normal(scale=10.0, size=(4, 4)) + 0.0001)

    d_hist, h_hist = coevolution_simulation(d, h, W)

    # d and h must always properly represent
    for d in d_hist:
        np.testing.assert_allclose(sum(d), 1.0)
    for h in h_hist:
        np.testing.assert_allclose(sum(h), 1.0)
