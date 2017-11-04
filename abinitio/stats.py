
import numpy as np
import pandas as pd


class IStoppingCondition(object):

    def handle_update(self, value):
        raise NotImplementedError

    def should_stop(self):
        raise NotImplementedError


class IterationStoppingCondition(IStoppingCondition):

    def __init__(self, max_iterations=100, **kwargs):
        super(IterationStoppingCondition, self).__init__(**kwargs)
        self._max_iterations = max_iterations
        self._iterations = 0

    def handle_update(self, value):
        self._iterations += 1

    def should_stop(self):
        return self._iterations > self._max_iterations


class DifferenceStoppingCondition(IStoppingCondition):
    def __init__(self, min_difference=1e-5, **kwargs):
        super(DifferenceStoppingCondition, self).__init__(**kwargs)
        self._min_difference = min_difference
        self._last_value = None
        self._difference = None

    def handle_update(self, value):
        if self._last_value is not None:
            diff = value - self._last_value
            self._difference = np.sqrt(np.dot(diff, diff))
        self._last_value = value

    def should_stop(self):
        if self._difference is not None:
            return self._difference < self._min_difference
        return False


class History(IterationStoppingCondition):

    def __init__(self, **kwargs):
        super(History, self).__init__(**kwargs)
        self._history = []

    @property
    def history(self):
        return pd.DataFrame(self._history)

    def handle_update(self, value):
        super(History, self).handle_update(value)
        self._history.append(value)


# TODO replace with a list of sub stopping conditions with an or or an and
class StoppingCondition(DifferenceStoppingCondition,
                        IterationStoppingCondition):

    def __init__(self, **kwargs):
        super(StoppingCondition, self).__init__(**kwargs)

    def handle_update(self, value):
        DifferenceStoppingCondition.handle_update(self, value)
        IterationStoppingCondition.handle_update(self, value)

    def should_stop(self):
        return IterationStoppingCondition.should_stop(self) or\
            DifferenceStoppingCondition.should_stop(self)


def sample_expectation(samples, g):
    """

    Parameters
    ----------
    samples : pandas.DataFrame
    g : callable

    Returns
    -------
    float
    """
    sample_values = samples.apply(g, axis=1)
    return sample_values.mean()


def fast_ica(samples, f, f_dot, stopping_condition=None):
    """

    Parameters
    ----------
    samples
    f
    f_dot
    stopping_condition : IStoppingCondition or None

    Returns
    -------

    """

    stopping_condition = stopping_condition or StoppingCondition()

    def _a(u):
        return sample_expectation(samples, lambda x: f_dot(np.dot(u, x))) * u

    def _b(u):
        return sample_expectation(samples, lambda x: f(np.dot(u, x)) * x)

    w = pd.Series([not i for i in range(len(samples.columns))])

    while not stopping_condition.should_stop():
        w = _a(w) - _b(w)
        w = w / np.sqrt(np.dot(w, w))

        if isinstance(w, pd.Series) and w[0] < 0.0:
            w = -1.0 * w

        stopping_condition.handle_update(w)

    return w, stopping_condition
