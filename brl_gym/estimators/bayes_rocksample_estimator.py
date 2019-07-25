import numpy as np
from gym.utils import seeding
from gym.spaces import Box

from brl_gym.estimators.estimator import Estimator
from brl_gym.wrapper_envs.env_sampler import EnvSampler
from brl_gym.envs.rocksample import RockSample, Action, StateType, Sense, load_env

def flatten_to_belief(belief_per_rock, cases_np):
    belief = []

    # Bernoulli distribution for each rock
    good_probs = [belief_per_rock ** case * (1 - belief_per_rock) ** (1 - case) for case in cases_np]

    # Each rock is independent, so multiply the probabilities
    belief = np.prod(np.array(good_probs), axis=1)

    # Should already sum to 1 but doing this just for numerical stability
    belief /= np.sum(belief)

    return belief

class BayesRockSampleEstimator(Estimator, EnvSampler):
    def __init__(self, num_rocks, observation_space, action_space,
        grid_size, start_coords, rock_positions,
        good_rock_probability=0.5):

        self.num_rocks = num_rocks
        self.num_cases = 2**self.num_rocks
        self.grid_size = grid_size
        self.start_coords = start_coords
        self.rock_positions = rock_positions
        self.good_rock_probability = good_rock_probability

        # ['0000', '0001', ..., '1111'], each corresponding to rock states
        # 0 is bad, 1 is good.
        self.cases = ['{{:0{}b}}'.format(self.num_rocks).format(x) \
                      for x in range(self.num_cases)]
        self.cases_np = [np.array([int(x) for x in case]) for case in self.cases]

        self.seed()
        self.reset()

        self.belief_low = np.zeros(self.belief.shape).ravel()
        self.belief_high = np.ones(self.belief.shape).ravel()

        self.param_low = np.zeros(self.belief.shape).ravel()
        self.param_high = np.ones(self.belief.shape).ravel()

        belief_space = Box(self.belief_low, self.belief_high, dtype=np.float32)

        super(BayesRockSampleEstimator, self).__init__(
            observation_space, action_space, belief_space)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def reset(self):
        self.belief = np.ones(self.num_rocks) * self.good_rock_probability

    def estimate(self, action, observation, **kwargs):
        """
        Given the latest (action, observation), update belief
        observation: [y, x, estimate1, ..., estimateK, accuracy1, ..., accuracyK,
                      distance1, ...distanceK]
        """

        if action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
            # No update
            return self.belief

        if not kwargs:
            self.reset()
            return self.belief

        coordinates, obs = kwargs['coordinates'], kwargs['estimate']

        accuracy = kwargs['sensor_accuracy']
        belief = np.copy(self.belief)

        if action == Action.SAMPLE:
            idx = kwargs['sampled_rock_idx']
            assert (idx is not None)
            if idx == -1:
                # SAMPLE was made on a location which doesn't contain a rock
                return self.belief
            else:
                # This position is no longer good.
                belief[idx] = 0
        else:
            idx = action - Action.SAMPLE - 1

            prior_good = belief[idx]

            if obs == StateType.GOOD_ROCK:
                p_obs = prior_good * accuracy + (1 - prior_good) * (1 - accuracy)
                p_obs_good_rock = prior_good * accuracy
            else:
                p_obs = prior_good * (1 - accuracy) + (1 - prior_good) * accuracy
                p_obs_good_rock = prior_good * (1 - accuracy)

            posterior = p_obs_good_rock / p_obs

            belief[idx] = posterior

        self.belief = belief

        return belief

    def get_belief(self):
        return self.belief.copy()

    def get_mle(self):
        return np.around(self.belief)

    def sample(self):
        r = self.np_random.uniform(size=self.belief.size)
        rock_states = (r <= self.belief).astype(np.uint8)
        env = RockSample(
            self.grid_size, self.num_rocks,
            self.start_coords,
            self.rock_positions,
            start_rock_state=rock_states)
        return env

if __name__ == "__main__":
    # Test
    env = load_env()
    estimator = BayesRockSampleEstimator(
        env.num_rocks, env.observation_space, env.action_space,
        env.grid_size, env.default_start_coords, env.rock_positions,
        env.good_rock_probability)

    print ("Rocks", env.start_rock_state)
    print ("")

    # Sense 1st rock
    action = Action.SAMPLE + 1
    obs, reward, done, info = env.step(action)
    print (obs, info)
    belief = estimator.estimate(action, obs, **info)
    print (estimator.get_belief())
    print ("MLE", estimator.get_mle())

    # Sense 1st rock again
    action = Action.SAMPLE + 1
    obs, reward, done, info = env.step(action)
    print ("")
    print (obs, info)
    belief = estimator.estimate(action, obs, **info)
    print (belief)
    print ("MLE", estimator.get_mle())

    # Go to first rock
    env.state[:2] = env.rock_positions[0]

    # Sample
    obs, reward, done, info = env.step(Action.SAMPLE)
    print ("")
    print (obs, info)
    belief = estimator.estimate(Action.SAMPLE, obs, **info)
    print (belief)
    print ("MLE", estimator.get_mle())

    # Sample env from current estimate
    sampled_env = estimator.sample()
    print("Env state", sampled_env.start_state)
