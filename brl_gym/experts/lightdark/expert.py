from brl_gym.experts.expert import Expert
from brl_gym.wrapper_envs.wrapper_lightdark import BayesLightDark

class LightDarKExpert(Expert):
    def __init__(self):
        env = BayesLightDark()
        obs_dim = env.env.observation_space.low.shape[0]
        bel_dim = env.estimator.belief_space.low.shape[0]
        super(LightDarKExpert, self).__init__(obs_dim, bel_dim)

    def action(self, inputs, infos=None):
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)

        obs, bel = self._split_inputs(inputs)

        # Ignore obs. Bel has all the necessary information
        dist_to_goal = bel[:, :2] * -1.0

        # Straight to the MLE goal
        return dist_to_goal

    def __call__(self, inputs):
        return self.action(inputs)
