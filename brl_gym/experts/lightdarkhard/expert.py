from brl_gym.experts.expert import Expert
from brl_gym.wrapper_envs.wrapper_lightdarkhard import BayesLightDarkHard
import numpy as np

class LightDarkHardExpert(Expert):
    def __init__(self):
        env = BayesLightDarkHard()
        obs_dim = env.env.observation_space.low.shape[0]
        bel_dim = env.estimator.belief_space.low.shape[0]
        super(LightDarkHardExpert, self).__init__(obs_dim, bel_dim)

    def action(self, inputs, infos=None):
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)

        obs, bel = self._split_inputs(inputs)
        goal = obs[:, 2:4]

        # Ignore obs. Bel has all the necessary information
        pos = bel[:, :2]
        dist_to_goal = goal - pos
        length = np.linalg.norm(dist_to_goal + 1e-3, axis=1)
        norm_dist = dist_to_goal / length.reshape(-1,1)

        dist_to_goal[length > 1.0] =  norm_dist[length > 1.0] * 2.0

        # Straight to the MLE goal
        return dist_to_goal

    def __call__(self, inputs):
        return self.action(inputs)

if __name__ == "__main__":
    import numpy as np
    env = BayesLightDarkHard()
    expert = LightDarkHardExpert()
    obs = env.reset()
    for _ in range(50):
        action = expert.action(obs)
        if True:#obs[-1] < 0.01:
            print('------------------')
            obs, r, d, info = env.step(action[0])
            print("obs", np.around(obs,2))
            print("action", action)
            print("x", env.env.x)
        else:
            obs, r, d, info = env.step(env.action_space.sample())

        if d:
            print("done")
            break

