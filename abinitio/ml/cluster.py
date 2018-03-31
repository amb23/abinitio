
import numpy as np


# TODO - move these to a generic folds
# {
def noop(*args, **kwargs):
    pass


def euclidean_distance(a, b):
    """

    Parameters
    ----------
    a : np.array
    b : np.array

    Returns
    -------
    float
    """
    return np.sum(np.power(a - b, 2))
# }


def lloyds_algorithm(k, points, distance_fn=euclidean_distance,
                     on_iteration=noop):
    """
    An algorithm that puts the data into a k-partition given by the distance
    provided. It is greedy and hence can be sensitive to initial conditions

    # TODO - add in multiple runs, initialisation conditions, mean generator
        (for non float arrays etc)

    Parameters
    ----------
    k : int
        the number of clusters to partition the data into
    points : np.array
        the points to be partitioned
    distance_fn : [optional] callable
        function to calculate the distance between two points
    on_iteration : [optional] callable
        function(iteration_count, points, partition) called when each iteration
        is performed. Defaults to noop.

    Returns
    -------
    dict<int, set<int>>
        The partition of the points
    """
    points = np.array(points)

    partition = _initial_partition(k, points)

    iterations = 0

    while True:
        changes = dict()
        means = _calculate_means(partition, points)
        for i, subset in partition.items():
            for j in subset:
                min_index = _calculate_min_index(points[j], means, distance_fn)

                if min_index != i:
                    # add this to the change set
                    assert j not in changes
                    changes[j] = (i, min_index)

        on_iteration(iterations, partition, points, )

        if not len(changes):
            # No changes here so break
            break
        # Apply the changes
        for j, (x, y) in changes.items():
            partition[x].remove(j)
            partition[y].add(j)

        iterations += 1

    return partition


def _initial_partition(k, points):
    partition = {i: set() for i in range(k)}
    for a in range(len(points)):
        partition[np.random.randint(0, k)].add(a)
    return partition


def _calculate_means(partition, points):
    return {i: np.sum(points[list(subset)], axis=0) / len(subset)
            for i, subset in partition.items()}


def _calculate_min_index(point, means, distance_fn):
    min_index = None
    min_dist = None

    for i, mean in means.items():
        if min_index is None:
            min_index = i
            min_dist = distance_fn(point, mean)
        else:
            dist = distance_fn(point, mean)
            if dist < min_dist:
                min_index = i
                min_dist = dist

    return min_index
