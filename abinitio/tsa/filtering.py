
import math
import numpy as np
import pandas as pd


def filter_data(data, lower_freq, upper_freq, method=None):
    """

    Parameters
    ----------
    data: pandas.DataFrame or pandas.Series
    lower_freq: float
    upper_freq: float
    method: [optional] str, Callable
        Defaults to TanhFilter
    Returns
    -------
    type(data)
        The filtered data
    """
    _f_data = _f_transform(data)  # TODO - use the index multiplier
    _method = method or TanhFilter(lower_freq, upper_freq)

    _filtered_f_data = _method(_f_data)

    return _inv_f_transform(_filtered_f_data, index=data.index)


class TanhFilter(object):

    def __init__(self, lower_freq, upper_freq):
        self._lf = lower_freq
        self._hf = upper_freq

        assert self._lf < self._hf

    def __call__(self, data):
        def weight(freq):
            return math.tanh(freq - self._lf) + math.tanh(self._hf - freq)

        _weight = pd.Series(data.index).apply(weight)

        return data.multiply(_weight, axis=0)


# TODO - set up single dispatch to handle pandas.Series
def _f_transform(data, index_multiplier=None):
    _f_data = pd.DataFrame(
        {column: np.fft.rfft(data[column]) for column in data.columns}
    )
    if index_multiplier is not None:
        _f_data.index *= index_multiplier
    return _f_data


# TODO - set up single dispatch to handle pandas.Series
def _inv_f_transform(data, index=None):
    """

    Parameters
    ----------
    data: pandas.DataFrame
    index: pandas.Index

    Returns
    -------
    pandas.DataFrame
    """
    _i_f_data = pd.DataFrame(
        {column: np.fft.irfft(data[column]) for column in data.columns}
    )
    if index is not None:
        assert len(_i_f_data.index) == len(index)
        _i_f_data.index = index
    return _i_f_data
