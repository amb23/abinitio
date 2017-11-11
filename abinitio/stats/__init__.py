
import numpy as np
import pandas as pd

from abinitio.stats import functional


def sample_expectation(samples, g):
    """
    Parameters
    ----------
    samples : pandas.DataFrame
    g : Callable<float(pandas.Series)> or
        Callable<pandas.Series(pandas.Series)>
    Returns
    -------
    float or pandas.Series
    """
    sample_values = samples.apply(g, axis=1)
    return sample_values.mean()


def fast_ica(samples, f, f_dot, **kwargs):
    """

    Parameters
    ----------
    samples
    f: Callable<float(float)>
    f_dot: Callable<float(float)>
        Derivative of the above algorithm
    kwargs: dict
        Passed onto functional.iteration_algorithm

    Returns
    -------

    """

    def _a(w):
        return sample_expectation(samples, lambda x: f_dot(np.dot(w, x))) * w

    def _b(w):
        return sample_expectation(samples, lambda x: f(np.dot(w, x)) * x)

    def update(w):
        w = _a(w) - _b(w)
        w = w / np.sqrt(np.dot(w, w))
        if isinstance(w, pd.Series) and w[0] < 0.0:
            w = -1.0 * w
        return w

    w0 = pd.Series([not i for i in range(len(samples.columns))])

    return functional.iteration_algorithm(update, w0, **kwargs)
