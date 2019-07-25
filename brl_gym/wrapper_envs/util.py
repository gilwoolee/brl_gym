import numpy as np

def to_one_hot(observation, dim):
    """
    Convert Discrete observation to one-hot vector
    """
    v = np.zeros(dim)
    v[observation] = 1
    return v

def from_one_hot(observation):
    assert (np.sum(observation) == 1)
    return np.argmax(observation)

def discount(x, gamma):
    """
    computes discounted sums along 0th dimension of x.

    inputs
    ------
    x: ndarray
    gamma: float

    outputs
    -------
    y: ndarray with same shape as x, satisfying

        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1

    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
