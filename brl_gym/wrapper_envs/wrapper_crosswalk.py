import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arrow
from matplotlib.patches import FancyArrowPatch, ArrowStyle

from brl_gym.wrapper_envs import BayesEnv
from brl_gym.envs.crosswalk_vel import CrossWalkVelEnv
from brl_gym.envs.crosswalk import CrossWalkEnv, colors
from brl_gym.estimators.bayes_crosswalk_estimator import BayesCrosswalkEstimator, get_angles

def get_pedestrian_directions(speeds, angles):
    return speeds.reshape(-1,1) \
            * np.array([-np.sin(angles),
                    np.cos(angles)]).transpose()

class BayesCrossWalkEnv(BayesEnv):
    def __init__(self, env_type="velocity", timestep=0.1):
        if env_type == "velocity":
            self.env = CrossWalkVelEnv(timestep=timestep)
        else:
            self.env = CrossWalkEnv()
        self.estimator = BayesCrosswalkEstimator()
        self.num_pedestrians = self.env.num_pedestrians
        self.nbins = self.estimator.nbins
        super(BayesCrossWalkEnv, self).__init__(self.env, self.estimator)

    def _expand_belief(self, obs, belief, **kwargs):
        # Convert belief into useful features

        if 'pedestrians' not in kwargs:
            peds = obs[-self.num_pedestrians*2*2:-self.num_pedestrians*2].reshape(-1, 2)
            directions = obs[-self.num_pedestrians*2:].reshape(-1,2) - peds
            speeds = np.linalg.norm(directions, axis=1)
        else:
            peds, speeds = kwargs['pedestrians'], kwargs['pedestrian_speeds']
        # expected_angles = np.vstack([
        #             get_angles(peds[:self.num_pedestrians // 2], self.estimator.GOALS_RIGHT),
        #             get_angles(peds[self.num_pedestrians // 2:], self.estimator.GOALS_LEFT)
        #             ])
        # weighted_angles = np.sum(expected_angles*belief, axis=1)

        weighted_goals = belief[:, self.nbins:self.nbins+2]
        diff = weighted_goals - peds
        direction = diff / np.linalg.norm(diff,axis=1).reshape(-1,1)
        weighted_directions = speeds.reshape(-1,1) * direction

        # # Stored for rendering
        # self.weighted_directions = weighted_directions
        # self.weighted_angles = weighted_angles
        # self.expected_angles = expected_angles
        # self.pedestrian_speeds = speeds
        # self.belief = belief

        obs[-self.num_pedestrians*2:] = (peds + weighted_directions).ravel()
        obs = np.concatenate([obs, belief.ravel()])
        return obs

    def reset(self):
        obs = self.env.reset()
        peds = obs[8:8+self.env.num_pedestrians*2].reshape(-1, 2)
        bel = self.estimator.estimate(None, obs, **{'pedestrians':peds})
        obs = self._expand_belief(obs, bel)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # estimate
        belief = self.estimator.estimate(action, obs, **info)
        obs = self._expand_belief(obs, belief, **info)
        return obs, reward, done, info

    def _visualize(self, filename=None, transparent=True):
        env = self.env
        fig, ax = plt.subplots(1)
        [p.remove() for p in reversed(ax.patches)]

        # Draw boundaries
        plt.plot([0, 0], [-10, 10], linewidth=5, color='k')
        plt.plot([9, 9], [-10, 10], linewidth=5, color='k')

        plt.axis('square')
        plt.axis('off')
        plt.xlim((-1, 10))
        plt.ylim((-6, 5))

        # Car
        # Pose of the end of the car
        car = env.x.copy()

        car = Rectangle((car[0], car[1]),
                        0.3, env.car_length, angle=np.rad2deg(env.angle), color=colors[-1])
        plt.gca().add_patch(car)

        # pedestrians
        pedestrians = env.pedestrians
        pedestrian_angles = self.weighted_directions
        pedestrian_directions = self.weighted_directions * 5.0
        angles = self.expected_angles
        belief = self.belief



        for i in range(len(pedestrians)):
            ped = Circle(pedestrians[i], radius=0.2, color=colors[i], zorder=20)
            plt.gca().add_patch(ped)

            style = ArrowStyle.Simple(head_length=1, head_width=1, tail_width=1)

            xy = pedestrians[i]
            for j in range(3):
                dxdy = get_pedestrian_directions(self.pedestrian_speeds[i], angles[i, j])[0] * 5.0
                arrow = Arrow(xy[0], xy[1], dxdy[0], dxdy[1], color=colors[i],
                        linewidth=0, width=0.3,
                        alpha=belief[i, j],
                        fill=True)
                # print(belief[i,j])
                # arrow = FancyArrowPatch(posA=xy, posB=xy+dxdy, arrowstyle='simple', color=colors[i], alpha=belief[i, j], linewidth=1.0)
                plt.gca().add_patch(arrow)

                arrow = Arrow(xy[0], xy[1], dxdy[0], dxdy[1],
                        edgecolor=colors[i], width=0.3,
                        linewidth=0.5, fill=False,
                        )
                plt.gca().add_patch(arrow)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', transparent=transparent)


class MLECrossWalkEnv(BayesCrossWalkEnv):
    def reset(self):
        obs = self.env.reset()
        self.estimator.estimate(None, obs)
        obs = self._expand_belief(obs, self.estimator.get_mle())
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # estimate
        self.estimator.estimate(action, obs, **info)
        obs = self._expand_belief(obs, self.estimator.get_mle(), **info)
        return obs, reward, done, info



if __name__ == "__main__":
    env = BayesCrossWalkEnv()
    obs = env.reset()

    for _ in range(5):
        obs, _, _, _ = env.step(env.action_space.sample())
        env._visualize("test.png", transparent=False)
        print(np.around(obs,1))


