import numpy as np
from scipy.special import logsumexp
from gym.utils import seeding
from gym.spaces import Box
from brl_gym.estimators.estimator import Estimator
from brl_gym.wrapper_envs.env_sampler import EnvSampler
from brl_gym.envs.chain import Chain, Action, State


class BayesChainEstimator(Estimator, EnvSampler):
    def __init__(self, slip_prob_a, slip_prob_b=None, semitied=False):
        """
        slip_prob_a: Array of possible slip probabiliies for action A
        slip_prob_b: Array of possible slip probabiliies for action B
        If tied, only slip_prob_a is used for both actions.
        """
        env = Chain()

        self.semitied = semitied

        if not semitied:
            self.slip_probs = np.array(slip_prob_a)
        else:
            if slip_prob_b is None:
                slip_prob_b = slip_prob_a
            self.slip_probs = np.stack([slip_prob_a, slip_prob_b])

        self.belief_low = np.zeros(self.slip_probs.shape).ravel()
        self.belief_high = np.ones(self.slip_probs.shape).ravel()

        self.belief_space = Box(self.belief_low, self.belief_high, dtype=np.float32)

        super(BayesChainEstimator, self).__init__(
            env.observation_space, env.action_space, self.belief_space)

        if not semitied:
            self.param_low = np.array([0])
            self.param_high = np.array([1])
        else:
            self.param_low = np.zeros(2)
            self.param_high = np.ones(1)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def reset(self):
        if not self.semitied:
            self.belief = np.ones(self.slip_probs.shape) / self.slip_probs.size
        else:
            self.belief = np.ones(self.slip_probs.shape) / self.slip_probs.shape[1]
        self.last_observation = State.ZERO
        return self.belief.ravel()

    def estimate(self, action, observation, **kwargs):
        """
        Given the latest (action, observation), update belief
        observation: {slip:True/False}
        """
        if 'slip' not in kwargs:
            return self.reset()
        slip = kwargs['slip']

        if not self.semitied:
            # P(obs) = P(obs|phi_1)P(phi_1) + ... P(obs|phi_K)P(phi_K)
            p_obs_per_mdp = self.slip_probs if slip else 1.0 - self.slip_probs
            p_obs_given_prior = p_obs_per_mdp * self.belief
            p_mdp_given_obs = p_obs_given_prior / np.sum(p_obs_given_prior)
            self.belief = p_mdp_given_obs
        else:
            p_obs_per_mdp = self.slip_probs[action] if slip else 1.0 - self.slip_probs[action]
            p_obs_given_prior = p_obs_per_mdp * self.belief[action]
            p_mdp_given_obs = p_obs_given_prior / np.sum(p_obs_given_prior)
            self.belief[action] = p_mdp_given_obs

        return self.get_belief()

    def get_belief(self):
        return self.belief.ravel()

    def get_mle(self):
        if self.semitied:
            indices = np.argmax(self.belief, axis=1)
            return np.array([
                self.slip_probs[0, indices[0]],
                self.slip_probs[1, indices[1]]])
        else:
            idx = np.argmax(self.belief)
            return np.array([self.slip_probs[idx]])

    def sample(self):
        if self.semitied:
            env = Chain(random_slip_prob=False,
                slip_prob_a=self.np_random.choice(self.slip_probs[0], p=self.belief[0]),
                slip_prob_b=self.np_random.choice(self.slip_probs[0], p=self.belief[1]),
                semitied=True)
        else:
            slip_prob = self.np_random.choice(self.slip_probs, p=self.belief)
            env = Chain(random_slip_prob=False,
                slip_prob_a=slip_prob, slip_prob_b=slip_prob,
                semitied=False)
        return env


if __name__ == "__main__":
    print("========== TIED ===========")
    # Tied
    env = Chain(random_slip_prob=False)
    state = env.reset()
    estimator = BayesChainEstimator(np.linspace(0,1,11))
    for _ in range(100):
        o, r, d, i = env.step(0)
        belief = estimator.estimate(0, o, **i)
    mle = estimator.get_mle()
    print (np.around(belief,2), "mle", mle)

    print("========== SEMI-TIED ===========")
    # SemiTied
    env = Chain(random_slip_prob=True, semitied=True)
    state = env.reset()
    estimator = BayesChainEstimator(np.linspace(0,1,11), np.linspace(0,1,11), semitied=True)
    for t in range(100):
        action = t % 2
        o, r, d, i = env.step(action)
        belief = estimator.estimate(action, o, **i)
    mle = estimator.get_mle()
    print (np.around(belief,2).reshape(2, -1), "mle", mle)


    print("========== SMAPLE ============")
    # Tied
    estimator = BayesChainEstimator(np.linspace(0,1,11))
    env = estimator.sample()
    print("Slip Probs:", env.slip_prob)

    # SemiTied
    estimator = BayesChainEstimator(np.linspace(0,1,11), np.linspace(0,1,11), semitied=True)
    env = estimator.sample()
    print("Slip Probs:", env.slip_prob)

    import IPython; IPython.embed()

