import numpy as np


def get_dataset(sd=0.1, num_points_per_cluster=1000):

    return np.concatenate([
        np.random.multivariate_normal([1, 1], [[sd, 0], [0, sd]], size=num_points_per_cluster),
        np.random.multivariate_normal([1, -1], [[sd, 0], [0, sd]], size=num_points_per_cluster),
        np.random.multivariate_normal([-1, 1], [[sd, 0], [0, sd]], size=num_points_per_cluster),
        np.random.multivariate_normal([-1, -1], [[sd, 0], [0, sd]], size=num_points_per_cluster)
    ], axis=0)
