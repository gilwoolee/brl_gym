import numpy as np
from brl_gym.estimators.estimator import Estimator
from gym.spaces import Box
from brl_gym.envs.mujoco.doors import DoorsEnv

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
    def __init__(self):
        env = DoorsEnv()

        self.num_doors = 4
        self.num_cases = 2**self.num_doors
        self.cases =  ['{{:0{}b}}'.format(self.num_doors).format(x) \
                      for x in range(self.num_cases)]
        self.cases_np = [np.array([int(x) for x in case]) for case in self.cases]

        self.belief_low = np.zeros(self.num_cases)
        self.belief_high = np.ones(self.num_cases)

        belief_space = Box(self.belief_low, self.belief_high, dtype=np.float32)


        super(BayesDoorsEstimator, self).__init__(
                env.observation_space, env.action_space, belief_space)

        self.reset()

    def reset(self):
        self.belief = np.ones(self.num_doors)*0.5
        self.flat_belief = flatten_to_belief(self.belief, self.cases_np)


    def estimate(self, action, observation, **kwargs):
        if not 'doors' in kwargs and 'collision' not in kwargs and 'pass_through' not in kwargs:
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

                posterior[i] = p_obs_open /  p_obs
            self.belief = posterior

        if 'collision' in kwargs:
            self.belief[kwargs['collision']] = 0.0
            print("door ", kwargs['collision'], 'collision')

        if 'pass_through' in kwargs:
            self.belief[kwargs['pass_through']] = 1.0
            print("door ", kwargs['pass_through'], 'pass_through')

        self.flat_belief = flatten_to_belief(self.belief, self.cases_np)
        return self.flat_belief

    def get_belief(self):
        return self.flat_belief


if __name__ == "__main__":
    # from brl_gym.envs.mujoco.doors import DoorsEnv
    # env = DoorsEnv()
    estimator = BayesDoorsEstimator()
    # env.reset()
    # estimator.reset()

    # for _ in range(10):
    #     action = np.array([0,0,1])
    #     o, r, d, info = env.step(action)
    #     belief = estimator.estimate(action, o, **info)
    #     print(info, estimator.belief)
    print(estimator.cases_np)
    print(estimator.cases)
    print(flatten_to_belief(np.array([0,0,0,1.0]), estimator.cases_np))
