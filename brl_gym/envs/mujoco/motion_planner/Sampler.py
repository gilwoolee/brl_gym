import numpy as np

class Sampler:
    def __init__(self, env):
        self.env = env
        self.xlimit = self.env.xlimit
        self.ylimit = self.env.ylimit

    def sample(self, num_samples):
        """
        Samples configurations.
        Each configuration is (x, y).

        @param num_samples: Number of sample configurations to return
        @return 2D numpy array of size [num_samples x 2]
        """
        x = np.random.choice(np.arange(self.xlimit[0], self.xlimit[1]), size=num_samples)
        y = np.random.choice(np.arange(self.ylimit[0], self.ylimit[1]), size=num_samples)
        return np.array([x, y]).transpose()
