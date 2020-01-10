from brl_gym.experts.expert import Expert
from brl_gym.wrapper_envs.mujoco.wrapper_wam_find_obj import BayesWamFindObj
import numpy as np

def get_action(dists_to_target):
    # Expert action: Get closer to target
    pos_ctrl = dists_to_target.copy()
    pos_ctrl = pos_ctrl / np.linalg.norm(pos_ctrl, axis=1).reshape(-1,1)
    pos_ctrl *= 0.01
    return pos_ctrl
    #rot = np.tile(np.array([1,0,0,0], dtype=np.float32), (dists_to_target.shape[0], 1))
    #return np.concatenate([pos_ctrl, rot], axis=1)

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
        action = get_action(obs)
        return action

    def __call__(self, inputs):
        return self.action(inputs)
