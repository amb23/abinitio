
import numpy as np
import pandas as pd


def uniform_dist(n=1000, d=1, rotation=None):
    sample = pd.DataFrame(np.random.rand(n, d)).subtract(
        pd.Series([0.5, 0.5]),
        axis=1
    )

    if d == 2 and isinstance(rotation, float) and rotation:
        a, b = np.cos(rotation), np.sin(rotation)
        _rotation = np.array([a, b], [-b, a])
        sample = sample.apply(lambda x: pd.Series(np.dot(_rotation, x)), axis=1)

    return sample
