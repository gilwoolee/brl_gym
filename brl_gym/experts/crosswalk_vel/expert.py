from brl_gym.experts.expert import Expert
from brl_gym.wrapper_envs.wrapper_crosswalk import BayesCrossWalkEnv
from brl_gym.envs.crosswalk_vel import CrossWalkVelEnv

import numpy as np

# Fixes direction as well
class CrossWalkVelExpert(Expert):
    def __init__(self, timestep=0.1, horizon=50):
        env = BayesCrossWalkEnv(env_type="velocity")
        obs_dim = env.env.observation_space.low.shape[0]
        bel_dim = env.estimator.belief_space.low.size
        action_space = env.env.action_space
        self.num_pedestrians = env.num_pedestrians
        self.car_y_goal = env.env.car_y_goal
        self.horizon = horizon

        super(CrossWalkVelExpert, self).__init__(obs_dim, bel_dim)

        velocities = np.linspace(action_space.low[0], action_space.high[0], 3, endpoint=True)
        steering_angles = np.linspace(action_space.low[1], action_space.high[1], 7, endpoint=True)

        self.params = np.array(np.meshgrid(velocities.reshape(-1,1), steering_angles.reshape(-1, 1))).transpose(1,2,0)
        self.timestep = timestep
        increments = np.arange(1, horizon + 1) * timestep
        angles = steering_angles.reshape(1,-1) * increments.reshape(-1,1)
        pose_increments = np.array([-np.sin(angles), np.cos(angles)])[:,:,:,None] * velocities * timestep
        self.poses = np.cumsum(pose_increments, axis=1)
        self.angles = angles[None, :, :]
        self.increments = increments
        self.x_limit = [0.0, 3.5]
        self.time_weight = np.arange(horizon + 10, 10, -1).astype(np.float).reshape(1,-1,1,1)
        self.time_weight /= np.sum(self.time_weight)
        # self.time_weight = 1
        self.steering_angles = steering_angles
        """
        ca = car_angle = np.deg2rad(30.0)
        transform = np.array([[np.cos(ca), -np.sin(ca)],[np.sin(ca), np.cos(ca)]])
        poses = np.tensordot(transform, poses, 1)

        from matplotlib import pyplot as plt
        plt.figure()
        for i in range(poses.shape[2]):
            for j in range(poses.shape[3]):
                plt.plot(poses[0][:,i, j], poses[1][:, i, j])
        plt.show()
        import sys; sys.exit(0)
        """

    def action(self, inputs, infos=None):
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        obss, bels = inputs[:, :-self.belief_dim], inputs[:, -self.belief_dim:]
        pedestrians = obss[:, -self.num_pedestrians*4:-self.num_pedestrians*2].reshape(inputs.shape[0], -1, 2)
        directions = obss[:, -self.num_pedestrians*2:].reshape(inputs.shape[0], -1, 2) - pedestrians

        car, car_front = obss[:, :2], obss[:, 2:4]
        car_speed, car_angle = obss[:, 6], obss[:, 7]

        # try various angles
        action = self._forward_simulate(pedestrians, directions, car,
                                        car_front, car_speed, car_angle)
        return action

    def _transform(self, poses, angle, position):
        transform = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        poses = np.array([np.tensordot(transform[:,:,i], poses, 1) + position[i].reshape(-1,1,1,1)
                                      for i in range(angle.shape[0])])
        return poses

    def _forward_simulate(self, peds, ped_dirs, car, car_front, car_speed, car_angle):
        car_poses = self._transform(self.poses, car_angle, car)
        car_fronts = self._transform(self.poses, car_angle, car_front)
        car_angles = self.angles + car_angle.reshape(-1, 1, 1)

        # Pedestrian positions in N steps
        peds = (peds[:,:,:,None] + ped_dirs[:,:,:,None] * self.increments)[:,:,:,:,None,None]
        distance = (np.linalg.norm(peds - car_poses[:,None,:,:,:,:], axis=2))
        front_distance = (1+np.linalg.norm(peds- car_fronts[:,None,:,:,:,:], axis=2))
        # front_distance[front_distance > 1.5] = 1e8
        # distance[distance > 1.5] = 1e8
        cost = np.sum(1.0/front_distance + 1.0/distance, axis=1) * 200.0 / self.horizon
        # cost = np.sum(1.0/distance**2, axis=1) * 50.0 / self.num_pedestrians

        # penalize distance to goal
        distance_to_goal = self.car_y_goal - car_poses[:, 1, :, : ,:]
        distance_to_goal[distance_to_goal < 0] = 0.0

        cost += np.abs(distance_to_goal) * 1.0
        cost += np.abs(car_angles)[:,:,:,None] * 5.0

        # Distance to sides
        distance_to_left = np.abs(self.x_limit[0] - car_poses[:, 0, :, :, :])
        distance_to_right = np.abs(self.x_limit[1] - car_poses[:, 0, :, : ,:])

        cost -= np.abs(distance_to_left) * 0.1
        cost -= np.abs(distance_to_right) * 0.1

        # time-weighted cost
        self.time_weight = 1
        cost = np.sum(cost*self.time_weight,axis=1) #/ self.horizon
        # Choose one with smallest cost
        bests = [np.unravel_index(np.argmin(c), c.shape) for c in cost]
        best_params = self.params[tuple(np.array(bests).T)]
        return best_params

if __name__ == "__main__":
    import cProfile, os
    # profile = cProfile.Profile()
    # profile.enable()
    rewards = []
    lengths = []
    all_rewards = []
    collision = 0
    for i in range(200):
        env = BayesCrossWalkEnv(env_type="velocity")
        obs = env.reset()
        expert = CrossWalkVelExpert()

        os.makedirs("img/trial{}".format(i))

        reward = 0
        for t in range(200):
            action = expert.action(np.array([obs]))
            obs, r, done, _ = env.step(action[0])
            env.env._visualize(filename="img/trial{}/test{}.png".format(i, t))
            reward += r
            #print(r)
            #env.render()
            if done:
                break
        if r <= 0:
            collision += 1
        lengths += [t]

        print("   -  ", i)
        print("reward", reward)
        print("length", t)
        rewards += [reward]
    print("No-Collision", 1 - collision / len(lengths))
    print("reward-mean", np.mean(np.array(rewards)))
    print("reward-ste ", np.std(np.array(rewards))/np.sqrt(len(rewards)))
    print("length-mean", np.mean(np.array(lengths)))
    print("length-ste ", np.std(np.array(lengths))/ np.sqrt(len(lengths)))
