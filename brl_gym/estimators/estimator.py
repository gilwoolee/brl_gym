class Estimator:
    def __init__(self, observation_space, action_space, belief_space):
        self._observation_space = observation_space
        self._action_space = action_space
        self.belief_space = belief_space

    def reset(self):
        raise NotImplementedError

    def estimate(self, action, observation, **kwargs):
        raise NotImplementedError

    def get_belief(self):
        raise NotImplementedError
