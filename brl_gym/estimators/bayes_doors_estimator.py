import numpy as np
from brl_gym.estimators.estimator import Estimator
from gym.spaces import Box
from brl_gym.envs.mujoco.doors import DoorsEnv
# from brl_gym.estimators.learnable_bf import LearnableBF

def flatten_to_belief(belief_per_door, cases_np):
    belief = []

    # Bernoulli distribution for each rock
    good_probs = [belief_per_door ** case * (1 - belief_per_door) ** (1 - case) for case in cases_np]

    # Each rock is independent, so multiply the probabilities
    belief = np.prod(np.array(good_probs), axis=1)

    # Should already sum to 1 but doing this just for numerical stability
    belief /= np.sum(belief)

    return belief

class BayesDoorsEstimator(Estimator):
    def __init__(self, estimate_disturbance=False, residual=None):
        env = DoorsEnv()

        self.num_doors = 4
        self.num_cases = 2**self.num_doors
        self.cases =  ['{{:0{}b}}'.format(self.num_doors).format(x) \
                      for x in range(self.num_cases)]
        self.cases_np = [np.array([int(x) for x in case]) for case in self.cases]

        self.belief_low = np.zeros(self.num_cases)
        self.belief_high = np.ones(self.num_cases)
        # self.belief_low = np.zeros(self.num_doors)
        # self.belief_high = np.ones(self.num_doors)
        self.param_low = np.zeros(self.num_doors)
        self.param_high = np.ones(self.num_doors)

        belief_space = Box(self.belief_low, self.belief_high, dtype=np.float32)
        self.param_space = Box(self.param_low, self.param_high)

        super(BayesDoorsEstimator, self).__init__(
                env.observation_space, env.action_space, belief_space)

        self._observaiton_space = env.observation_space
        self._action_space = env.action_space

        self.reset()

    def reset(self):
        self.belief = np.ones(self.num_doors)*0.5
        # self.flat_belief = flatten_to_belief(self.belief, self.cases_np)
        # return self.flat_belief
        return self.belief

    def estimate(self, action, observation, **kwargs):
        if not 'doors' in kwargs and 'collision' not in kwargs and 'pass_through' not in kwargs:
            # return self.flat_belief
            return self.belief

        if 'doors' in kwargs:
            doors = kwargs['doors']
            accuracy = kwargs['accuracy']

            prior = self.belief
            posterior = self.belief.copy()
            for i, d in enumerate(doors):
                if d == True: # open
                    p_obs = prior[i] * accuracy[i] + (1 - prior[i]) * (1 - accuracy[i])
                    p_obs_open = prior[i] * accuracy[i]
                else:
                    p_obs = prior[i] * (1 - accuracy[i]) + (1 - prior[i]) * accuracy[i]
                    p_obs_open = prior[i] * (1 - accuracy[i])

                posterior[i] = p_obs_open /  (p_obs + 1e-10)
            self.belief = posterior

        if 'collision' in kwargs:
            self.belief[kwargs['collision']] = 0.0
            # print("door ", kwargs['collision'], 'collision')

        if 'pass_through' in kwargs:
            self.belief[kwargs['pass_through']] = 1.0
            # print("door ", kwargs['pass_through'], 'pass_through')

        # self.flat_belief = flatten_to_belief(self.belief, self.cases_np)
        # return self.flat_belief
        return self.belief

    def get_belief(self):
        # return self.flat_belief
        return self.belief

    def get_mle(self):
        return np.around(self.belief)

    def flatten_to_belief(self, belief):
        return flatten_to_belief(belief, self.cases_np)

    def get_flat_belief(self):
        return self.flatten_to_belief(self.belief)


# class LearnableDoorsBF(LearnableBF, BayesDoorsEstimator):
#     def __init__(self):

#         self.belief_dim = 4
#         BayesDoorsEstimator.__init__(self)

#         self.belief_low = np.zeros(self.num_doors)
#         self.belief_high = np.ones(self.num_doors)
#         self.belief_space = Box(self.belief_low, self.belief_high, dtype=np.float32)

#         LearnableBF.__init__(self, self._action_space,
#             self._observaiton_space, self.belief_space, nonlinear='relu', normalize=False)

#     def reset(self):
#         return LearnableBF.reset(self)

#     def estimate(self, action, observation, **kwargs):
#         return LearnableBF.estimate(self, action, observation, **kwargs)



if __name__ == "__main__":
    # from brl_gym.envs.mujoco.doors import DoorsEnv
    # env = DoorsEnv()
    estimator = LearnableDoorsBF()
    env = DoorsEnv()

    o = env.reset()
    for _ in range(100):
        a = env.action_space.sample()
        o, r, d, info = env.step(a)
        info['done'] = d
        b = estimator.estimate(a, o, **info)

        print(b)

    import IPython; IPython.embed()