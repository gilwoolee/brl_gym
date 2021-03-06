from brl_gym.experts.expert import Expert
from brl_gym.wrapper_envs.wrapper_lightdark import BayesLightDark
import numpy as np

class LightDarkExpert(Expert):
    def __init__(self):
        env = BayesLightDark()
        obs_dim = env.env.observation_space.low.shape[0]
        bel_dim = env.estimator.belief_space.low.shape[0]
        super(LightDarkExpert, self).__init__(obs_dim, bel_dim)

    def action(self, inputs, infos=None):
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)

        obs, bel = self._split_inputs(inputs)
        # Ignore obs. Bel has all the necessary information
        dist_to_goal = bel[:, :2]
        #dist_to_goal /= np.linalg.norm(dist_to_goal, axis=1).reshape(-1,1)
        #dist_to_goal *= 0.5

        # Straight to the MLE goal
        return dist_to_goal

    def __call__(self, inputs):
        return self.action(inputs)
if __name__ == "__main__":
    import numpy as np
    env = BayesLightDark()
    expert = LightDarkExpert()
    obs = env.reset()
    for _ in range(100):
        print()
        action = expert.action(obs)
        print("goal  ", env.env.goal)
        print("self  ", env.env.x)
        print("dir   ", np.around(env.env.goal - env.env.x,2))
        print("obs   ", np.around(obs,2))
        print("action", np.around(action,2))
        obs, _, _, info = env.step(action[0])

