from .maze import MotionPlanner
import numpy as np
import pickle
import os
from .util import convert_3D_to_2D, convert_2D_to_3D


def get_closest_point(waypoints, position):
    if waypoints is None or waypoints is False:
        raise RuntimeError
    waypoints = waypoints.reshape(-1, 2)
    dist = np.linalg.norm(waypoints - position, axis=1)
    idx = np.argmin(dist)
    return idx


# Copied from wrapper_maze
def simple_expert_actor(mp, pose, target):
    start = pose[:2]
    waypoints = mp.motion_plan(start, target, reuse=False, shortcut=True)

    if not isinstance(waypoints, np.ndarray) and (waypoints == False or waypoints is None):
        print("Failed from ", start, "to", target)
        raise RuntimeError("No waypoint")
        return None

    # lookahead = 0
    # idx = min(get_closest_point(waypoints, pose[:2]) + lookahead, waypoints.shape[0]-1)

    idx = min(5, len(waypoints) - 1)
    direction = waypoints[idx] - pose[:2]
    # if np.linalg.norm(direction) < 0.01:
    # print("direction", direction)
    # import IPython; IPython.embed()

    direction /= (np.linalg.norm(direction))
    return direction

    # import IPython; IPython.embed(); import sys; sys.exit(0)
    # print(pose, direction)
    while True:
        # Add noise
        noise = np.random.normal(size=(100,2))
        noise /= np.linalg.norm(noise, axis=1).reshape(-1, 1)
        noise *= 0.1

        directions = direction.copy() + noise

        # check for collision
        step_forward = start + directions * 0.01
        idx = np.all(step_forward < 1.5, axis=1)
        step_forward = step_forward[idx]
        directions = directions[idx]
        idx = np.all(step_forward > -1.5, axis=1)
        step_forward = step_forward[idx]
        directions = directions[idx]

        collision_free = mp.state_validity_checker(step_forward, use_sampling_map=True)
        directions = directions[collision_free]

        if len(directions) == 0:
            continue

        # choose the best direction
        dist = np.linalg.norm(direction - directions, axis=1)

        direction = directions[np.argmin(dist)]

        return direction / np.linalg.norm(direction)

# For a target, create a vectorfield across all points in the map so that there's no need to replan.
class VectorField:
    def __init__(self, make_new=False, target=0, maze_type=4):

        self.target_id = target
        self.vf = None
        self.maze_type = maze_type
        if self.maze_type == 4:
            from brl_gym.envs.mujoco.point_mass import GOAL_POSE
        else:
            from brl_gym.envs.mujoco.maze10 import GOAL_POSE
        self.goal = GOAL_POSE[target]

        dir_path = os.path.dirname(os.path.realpath(__file__))

        filename = os.path.join(dir_path, "resource", "vf_maze{}_{}.pkl".format(
            self.maze_type, target))
        self.planner = MotionPlanner(maze_type=maze_type, make_new=False)

        if make_new or not os.path.exists(filename):
            self.vf = self._make_vectorfield()
            self.visualize()
            pickle.dump(self.vf, open(filename, 'wb'))

        else:
            print("Load ", filename)
            self.vf = pickle.load(open(filename, 'rb'))


    def _make_vectorfield(self):
        vector_field = []

        if self.maze_type == 4:
            increment = 0.1
            xmin = 1.45
        else:
            increment = 0.05
            xmin = 1.40

        for x in np.arange(-xmin, xmin, increment):
            for y in np.arange(-xmin, xmin, increment):
        # while True:
            # while True:
                # For every (x, y) plan to target
                # x, y = -0.7, -1.40
                start = np.array([x, y])

                # Check if in collision
                collision_free = self.planner.state_validity_checker(
                    start, use_sampling_map=True)

                if not (collision_free):
                    # raise RuntimeError
                    continue

                direction = simple_expert_actor(self.planner, start, self.goal)

                if direction is None:
                    raise RuntimeError("No path from {} to {}".format(start, self.goal))

                vector_field += [np.concatenate([start, self.goal, direction], axis=0)]

                # print(direction)
                # import sys; sys.exit(0)

        return np.array(vector_field)


    def motion_plan(self, start, target):
        assert self.vf is not None
        assert np.all(target == self.goal)

        # Find min-dist collision-free point in vectorfield
        vf_st = self.vf[:, :2]
        st = np.tile(start, (vf_st.shape[0], 1, 1))
        vf_st = np.tile(vf_st, (1,1,1)).transpose(1,0,2)
        dist = np.linalg.norm(vf_st - st, axis=2)
        min_dist_idx = np.argmin(dist, axis=0)

        original_direction = self.vf[min_dist_idx, -2:]
        direction = 0.5*original_direction + 0.5*(self.vf[min_dist_idx, :2] - start)
        direction /= np.linalg.norm(direction, axis=1).reshape(-1, 1)

        return direction

    def state_validity_checker(self, config, use_sampling_map):
        return self.planner.state_validity_checker(config, use_sampling_map=use_sampling_map)

    def visualize(self):
        from matplotlib import pyplot as plt
        vf = self.vf
        fig, ax = plt.subplots()

        for row in vf:
            plt.arrow(row[0], row[1], row[4]*0.08, row[5]*0.08, shape='full', head_width=0.015, head_length=0.05)

        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])

        ax.set_aspect('equal')
        circle1 = plt.Circle((self.goal[0], self.goal[1]), 0.02, color='r')
        ax.add_artist(circle1)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.savefig(os.path.join(dir_path, "resource/vf_maze{}_{}.png".format(self.maze_type,
            self.target_id)))
        print("Saved fig for {}".format(self.target_id))
        # plt.show()
