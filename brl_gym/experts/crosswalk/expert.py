from brl_gym.experts.expert import Expert
from brl_gym.wrapper_envs.wrapper_crosswalk import BayesCrossWalkEnv
from brl_gym.envs.crosswalk import CrossWalkEnv

import numpy as np

class CrossWalkExpert(Expert):
    def __init__(self):
        env = BayesCrossWalkEnv()
        obs_dim = env.env.observation_space.low.shape[0]
        bel_dim = env.estimator.belief_space.low.size
        self.num_pedestrians = env.num_pedestrians

        super(CrossWalkExpert, self).__init__(obs_dim, bel_dim)

        self.dangle = (np.linspace(-1, 1, 5) * 0.1).reshape(5,1)
        self.accels = (np.linspace(-1.8,1.8,9) * 0.1).reshape(9,1)
        self.params = np.array(np.meshgrid(self.dangle, self.accels)).transpose(1,2,0)

    def action(self, inputs, infos=None):
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        obss, bels = inputs[:, :-self.belief_dim], inputs[:, -self.belief_dim:]
        pedestrians = obss[:, -self.num_pedestrians*4:-self.num_pedestrians*2].reshape(inputs.shape[0], -1, 2)
        directions = obss[:, -self.num_pedestrians*2:].reshape(inputs.shape[0], -1, 2) - pedestrians
        cars, car_fronts = obss[:, :2], obss[:, 2:4]
        car_speeds, car_angles = obss[:, 6], obss[:, 7]

        # try various angles
        action = self._forward_simulate(pedestrians, directions, cars[:,:,None,None,None],
                                        car_fronts[:,:,None,None,None], car_speeds, car_angles)
        return action * 10

    def _forward_simulate(self, peds, ped_dirs, cars, car_fronts, car_speeds, car_angles):
        dangle, accels, params = self.dangle, self.accels, self.params

        angles = np.tile(car_angles, (5, 1)) + dangle
        delta = np.array([-np.sin(angles), np.cos(angles)])[None,:,:,:]
        speeds = (np.tile(car_speeds, (9, 1)) + accels)[:,None,None,:]
        delta = (delta * speeds).transpose(3,1,0,2)

        cost = np.zeros((peds.shape[0], 9, 5)) + np.abs(angles).transpose()[:,None,:] * 2
        cost += -speeds.reshape(9, -1).transpose()[:,:,None] * 30

        increments = (np.arange(15) + 1.0)[None,None,None]
        ped_dirs = ped_dirs[:,:,:,None] * increments
        peds = peds[:,:,:,None] + ped_dirs

        delta = delta[:,:,:,:,None]*increments[:,:,:,None]
        cars = cars + delta

        dists = np.linalg.norm(peds[:,:,:,None,None,:] - cars[:,None,:,:,:,:], axis=2)

        car_fronts = car_fronts + delta
        front_dists = np.linalg.norm(peds[:,:,:,None,None,:] - car_fronts[:,None,:,:,:,:], axis=2)
        cost += np.sum(np.sum(1.0/dists**2  + 1.0/front_dists**2, axis=1), axis=3)

        # Choose one with smallest cost
        bests = [np.unravel_index(np.argmin(c), c.shape) for c in cost]
        best_params = params[tuple(np.array(bests).T)][:,[1,0]]

        return best_params

if __name__ == "__main__":

    rewards = np.zeros(100)
    for i in range(100):
        env = BayesCrossWalkEnv()
        obs = env.reset()
        expert = CrossWalkExpert()
        for _ in range(60):
            obs, r, done, _ = env.step(expert.action(np.array([obs,obs,obs]))[0])
            rewards[i] += r
            env.render()
            if done:
                break
        print(rewards[i])
    import IPython; IPython.embed()
    print(np.mean(rewards), np.std(rewards))
