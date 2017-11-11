"""
This module builds out some of the core functional tools we use throughout the
rest of the repo
"""

import numpy as np


class IStoppingCondition(object):
    """
    IStoppingCondition

    This class is used for determining when a stopping condition has been
    reached.    Implementations can be found below
    """

    def handle_update(self, value):
        """

        Parameters
        ----------
        value: numpy.array
            Most recent value created from the algorithm

        Returns
        -------

        """
        raise NotImplementedError

    def should_stop(self):
        """

        Returns
        -------
        bool
            Whether the condition has been satisfied
        """
        raise NotImplementedError


class IterationStoppingCondition(IStoppingCondition):
    """
    IterationStoppingCondition

    Stopping condition is satisfied when a certain number of iterations have
    reached
    """

    def __init__(self, max_iterations=100):
        self._max_iterations = max_iterations
        self._iterations = 0

    def handle_update(self, value):
        self._iterations += 1

    def should_stop(self):
        return self._iterations > self._max_iterations


class DifferenceStoppingCondition(IStoppingCondition):
    """
    DifferenceStoppingCondition

    Stopping condition is satisfied when the difference between the current
    value and the last is below a threshold
    """

    def __init__(self, min_difference=1e-5):
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


class MultiStoppingCondition(IStoppingCondition):
    """
    MultiStoppingCondition

    Contains a selection of stopping conditions. It can be constructed so that
    either any or all of them should have stopped for this condition to have
    been reached
    """

    def __init__(self, conditions, when='all'):
        """

        Parameters
        ----------
        conditions: list<IStoppingCondition>
            The sub conditions that go into this multi condition
        when: {'all', 'any'}
            If 'all' then all sub conditions must be satisfied for the stop,
            if 'any' then only one must be.
        """
        self._conditions = conditions
        self._when = getattr(np, when)

    def handle_update(self, value):
        for condition in self._conditions:
            condition.handle_update(value)

    def should_stop(self):
        sub_should_stop = [
            condition.should_stop() for condition in self._conditions
        ]
        return self._when(sub_should_stop)


class StoppingCondition(MultiStoppingCondition):
    """
    Default stopping condition stops after a number of iterations or if the
    difference is suitably small
    """

    def __init__(self, max_iterations=100, min_difference=1e-5):
        super(StoppingCondition, self).__init__(
            conditions=[
                IterationStoppingCondition(max_iterations=max_iterations),
                DifferenceStoppingCondition(min_difference=min_difference)
            ],
            when='any'
        )


def iteration_algorithm(update, initial_condition, stopping_condition=None):
    """

    Parameters
    ----------
    update: Callable<numpy.array(numpy.array)>
    initial_condition: numpy.array
    stopping_condition: IStoppingCondition or None
        defaults to StoppingCondition
    Returns
    -------
    numpy.array
        The result whn the stopping condition has been met
    """
    stopping_condition = stopping_condition or StoppingCondition()

    state = initial_condition

    while not stopping_condition.should_stop():
        state = update(state)
        stopping_condition.handle_update(state)

    return state
