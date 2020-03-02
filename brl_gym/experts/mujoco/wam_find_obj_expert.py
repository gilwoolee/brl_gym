from brl_gym.experts.expert import Expert
from brl_gym.wrapper_envs.mujoco.wrapper_wam_find_obj import BayesWamFindObj
import numpy as np

def get_action(hand_pos, obj_pos, top_shelf, bottom_shelf):
    dist = obj_pos - hand_pos
    pos_ctrl = dist.copy()
    pos_ctrl = pos_ctrl / np.linalg.norm(pos_ctrl, axis=1).reshape(-1,1)

    # Avoid colliding with shelf or can
    can_on_top = np.logical_and(obj_pos[:,2] > top_shelf[2],
                                hand_pos[:,2] < top_shelf[2] + 0.1)
    pos_ctrl[can_on_top, 2] += 1

    can_on_bottom = np.logical_and(obj_pos[:,2] < top_shelf[2],
                                   hand_pos[:, 2] < bottom_shelf[2])
    pos_ctrl[can_on_bottom, 2] += 1
    left_of_can = np.logical_and(hand_pos[:, 1] > obj_pos[:, 1] + 0.01,
                                 dist[:, 0] < 0.15)
    pos_ctrl[left_of_can, 0] = 0.0
    return pos_ctrl

class WamFindObjExpert(Expert):
    def __init__(self):
        env = BayesWamFindObj()
        obs_dim = env.env.observation_space.low.shape[0]
        bel_dim = env.estimator.belief_space.low.shape[0]
        super(WamFindObjExpert, self).__init__(obs_dim, bel_dim)

    def action(self, inputs, infos=None):
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)

        obs, bel = self._split_inputs(inputs)
        hand, top, bottom = obs[:, :3], obs[:, 6:9], obs[:, 9:12]
        obj = bel[:, :3]

        action = get_action(hand, obj, top[0], bottom[0])
        action = np.concatenate([action, np.zeros((action.shape[0],1))], axis=1)
                                #np.random.normal(size=action.shape[0]).reshape(-1,1)], axis=1)
        return action

    def __call__(self, inputs):
        return self.action(inputs)

if __name__ == "__main__":
    rewards = np.zeros(500)
    for i in range(500):
        expert = WamFindObjExpert()
        env = BayesWamFindObj()
        obs = env.reset()

        for _ in range(200):
            action = expert.action(obs.reshape(1, -1))[0]
            action[-1] = np.random.normal()
            obs, r, d, _ = env.step(action)
            #env.render()
            if d:
                break
            rewards[i] += r
        print(i, rewards[i])

    print(np.mean(rewards), np.std(rewards)/np.sqrt(len(rewards)))
