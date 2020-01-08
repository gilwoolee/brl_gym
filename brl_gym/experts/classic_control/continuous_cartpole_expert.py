from brl_gym.experts.expert import Expert
from brl_gym.wrapper_envs.classic_control.wrapper_continuous_cartpole import BayesContinuousCartPoleEnv
from brl_gym.envs.classic_control.continuous_cartpole import LQRControlCartPole
import numpy as np

class ContinuousCartPoleExpert(Expert):
    def __init__(self):
        env = BayesContinuousCartPoleEnv()
        obs_dim = env.env.observation_space.low.shape[0]
        bel_dim = env.estimator.belief_space.low.shape[0]
        super(ContinuousCartPoleExpert, self).__init__(obs_dim, bel_dim)

        envs = env.estimator.envs
        self.experts = [LQRControlCartPole(e) for e in envs]
        self.belief_threshold = 0.05 # Ignore belief lower than this

    def action(self, inputs, infos=None):
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        obss, bels = inputs[:, :4], inputs[:, 4:]

        act = np.zeros((inputs.shape[0], 1), dtype=np.float32)
        bels[bels < self.belief_threshold] = 0
        bels /= np.sum(bels, axis=1).reshape(-1, 1)
        for i in np.arange(obss.shape[0]):
            obs, bel = obss[i], bels[i]
            for j, b in enumerate(bel):
                if b == 0:
                    continue
                act[i] += self.experts[j].lqr_control(obs)[0] * b

        return act


if __name__ == "__main__":
    env = BayesContinuousCartPoleEnv()
    expert = ContinuousCartPoleExpert()

    obs = env.reset()
    for t in range(100):
        action = expert.action(obs)
        print(action)
        obs, _, d, _ = env.step(action[0])
        print(np.around(obs[4:],2))
        if d:
            print(t)
            break
        #env.render()


