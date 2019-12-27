from brl_gym.experts.expert import Expert
from brl_gym.wrapper_envs.wrapper_lightdark import BayesLightDark

class LightDarKExpert(Expert):
    def __init__(self):
        env = BayesLightDark()
        obs_dim = env.env.observation_space.n
        bel_dim = env.estimator.belief_space.n
        super(LightDarKExpert, self).__init__(self, obs_dim, bel_dim)

    def action(self, inputs, infos):
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)

        obs, bel = self.split_inputs(inputs)

        # Ignore obs. Bel has all the necessary information
        dist_to_goal = bel[:, :2]

        # Straight to the MLE goal
        return dist_to_goal

