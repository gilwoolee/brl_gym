from brl_gym.wrapper_envs.bayes_env import BayesEnv
import numpy as np


class DisturbanceLearner:
    def __init__(self, env):
        assert isinstance(env, BayesEnv)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.init_belief = env.estimator.reset()
        self.envs = env.estimator.envs

    def compute_error(self, trajectory):
        """
        Returns Disturbance instance
        """
        errors = np.zeros((len(self.envs), len(trajectory), len(trajectory[0][0])))
        states = np.zeros((len(trajectory), len(trajectory[0][0])))
        actions = np.zeros((len(trajectory), self.action_space.low.shape[0]))
        nstates = np.zeros((len(trajectory), len(trajectory[0][0])))

        for i, (s, a, next_s) in enumerate(trajectory):
            states[i] = s
            actions[i] = a
            nstates[i] = next_s
            for j, env in enumerate(self.envs):
                env.reset()
                env.set_state(s)
                env.step(a)
                predicted = env.get_state()
                errors[j, i] = next_s - predicted

        weighted_errors = np.sum(errors * self.init_belief[:, None, None], axis=0)
        return weighted_errors, states, actions, nstates

    def learn(self, trajectory):
        errors, states, actions, nstates = self.compute_error(trajectory)
        sa = np.concatenate([states, actions, np.ones((len(states),1))], axis=1)
        x = np.dot(np.linalg.pinv(sa), errors)
        mean_error = np.mean(np.dot(sa, x) - errors, axis=0)
        print("Mean error between the fit model and error", np.around(mean_error, 3))
        self.coeffs = x
        f = lambda s, a: self.learned_residual(s, a, x)
        return f

    def learned_residual(self, state, action, coeffs):
        print("residual")
        if isinstance(action, np.ndarray) or action is not None:
            print(np.dot(np.hstack([state, action, 1]), coeffs))
            return np.dot(np.hstack([state, action, 1]), coeffs)
        else:
            return np.dot(np.hstack([state, np.zeros(self.action_space.low.shape[0]), 1]), coeffs)


if __name__== "__main__":
    from brl_gym.wrapper_envs.classic_control.wrapper_continuous_cartpole import BayesContinuousCartPoleEnv

    env = BayesContinuousCartPoleEnv(noisy=False)
    learner = DisturbanceLearner(env)

    # Test env
    env = BayesContinuousCartPoleEnv(noisy=True)
    env.reset()
    print(env.env.ctrl_noise_scale)
    state = env.get_state()
    trajectory = []
    while True:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        nstate = env.get_state()
        trajectory += [(state, action, nstate)]
        state = nstate
        if done:
            break
    learned_residual = learner.learn(trajectory)


    # Bayes_env with learned dynamics
    env = BayesContinuousCartPoleEnv(noisy=False, learned_residual=learned_residual)
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)

        if done:
            break

    # TODO:
    # 0. Fix BayesContinuousCartpoleEstimator to use residual dynamics in its param envs.
    #    Verify that this results in more accurate bayes filter
    # 1. Collect data with  BayesContinuousCartPoleEnv(noisy=True)
    # 2. use BayesContinuousCartPoleEnv(noisy=False, learned_residual=learned_residual) to train a new agent.