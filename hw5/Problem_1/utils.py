import numpy as np
from sklearn.datasets import make_circles, make_moons, make_blobs
import os


def generate_data_lin(N=1000):
    x, y = make_blobs(N, centers=np.array([[-1, -1], [1, 1]]), n_features=2, cluster_std=0.2)
    y[y == 0] = -1

    return x, np.expand_dims(y, axis=1)


def generate_data_basis(P='circle', N=1000):
    if P == 'circle':
        phi = np.random.randn(N) * 2 * np.pi
        x1 = np.cos(phi)
        x2 = np.sin(phi)

        y = np.ones((N, 1))
        y[((x1 < 0) & (x2 > 0)) | ((x1 > 0) & (x2 < 0))] = -1

        x = np.stack((x1, x2), axis=1) + np.random.rand(N, 2) * 0.05

        return x, y

    if P == 'inner_circle':
        x, y = make_circles(N, factor=0.2, noise=0.1)
        y[y == 0] = -1

        return x, np.expand_dims(y, axis=1)

    if P == 'moons':
        x, y = make_moons(N, noise=.05)
        y[y == 0] = -1

        return x, np.expand_dims(y, axis=1)


def generate_data_non_lin(N=1000):
    x = np.random.randn(N,2)
    w = np.array([-.00, 0.01, 0.1, -0.04, 0.09, 0.02])
    features = np.hstack([np.ones([N,1]), x, x**2, x[:,:1]*x[:,1:2]])
    f = np.dot(features, w)
    labels = 2*((f + np.random.randn(N)*0.02)>0) - 1
    y = np.expand_dims(labels, axis=-1)
    return x, y


def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try: 
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise
