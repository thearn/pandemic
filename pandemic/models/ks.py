import numpy as np

def KS(g, rho=50.0):
    """
    Kreisselmeier-Steinhauser constraint aggregation function.
    """
    g_max = np.max(np.atleast_2d(g), axis=-1)[:, np.newaxis]
    g_diff = g - g_max
    exponents = np.exp(rho * g_diff)
    summation = np.sum(exponents, axis=-1)[:, np.newaxis]

    KS = g_max + 1.0 / rho * np.log(summation)

    dsum_dg = rho * exponents
    dKS_dsum = 1.0 / (rho * summation)
    dKS_dg = dKS_dsum * dsum_dg

    dsum_drho = np.sum(g_diff * exponents, axis=-1)[:, np.newaxis]
    dKS_drho = dKS_dsum * dsum_drho

    return KS, dKS_dg.flatten()