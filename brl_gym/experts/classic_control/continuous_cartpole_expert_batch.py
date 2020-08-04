from brl_gym.experts.expert import Expert
from brl_gym.wrapper_envs.classic_control.wrapper_continuous_cartpole import BayesContinuousCartPoleEnv
from brl_gym.envs.classic_control.continuous_cartpole import LQRControlCartPole
from brl_gym.envs.classic_control.lqr_batch import LQRControlCartPoleBatch
import numpy as np

class ContinuousCartPoleExpert(Expert):
    def __init__(self, nbatch=1):
        env = BayesContinuousCartPoleEnv()
        obs_dim = env.env.observation_space.low.shape[0]
        bel_dim = env.estimator.belief_space.low.shape[0]
        super(ContinuousCartPoleExpert, self).__init__(obs_dim, bel_dim)

        envs = env.estimator.envs
        self.nenvs = len(envs)
        self.experts = LQRControlCartPoleBatch(envs, nbatch=nbatch)
        self.belief_threshold = 0.05 # Ignore belief lower than this

    def action(self, inputs, infos=None):
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        obss, bels = inputs[:, :4], inputs[:, 4:]

        bels[bels < self.belief_threshold] = 0
        bels /= np.sum(bels, axis=1).reshape(-1, 1)
        actions = np.sum(self.experts.lqr_control(obss, bels) * bels, axis=1).reshape(-1,1)
        return actions


if __name__ == "__main__":
    env = BayesContinuousCartPoleEnv()
    expert = ContinuousCartPoleExpert()

    rewards = np.zeros(100)
    for i in range(1):
        print(i, )
        obs = env.reset()
        for t in range(500):
            action = expert.action(obs)[0]
            print(action)
            obs, r, d, _ = env.step(action)
            rewards[i] += r
            env.render()
            if d:
                break
        print(rewards[i])
            #env.render()

    print(np.mean(rewards), np.std(rewards) / np.sqrt(len(rewards)))
