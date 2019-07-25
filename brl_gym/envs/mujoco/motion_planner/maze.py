import numpy as np

import gym
from gym.spaces import Box
from gym import utils
from gym.utils import seeding
from .MapEnvironment import MapEnvironment
from .graph_maker import load_graph, add_node, make_graph
import os
import networkx as nx
from .astar import astar_path
from .Sampler import Sampler

# 2D mazemap version of point_mass.xml
mazemap = np.zeros((300, 300))
mazemap[0:5, :] = 1
mazemap[-5:, :] = 1
mazemap[:, 0:5] = 1
mazemap[:, -5:] = 1
mazemap[149:151,49:251] = 1
mazemap[199:201,199:301] = 1
mazemap[249:251,49:251] = 1
mazemap[49:51, 49:251] = 1
mazemap[99:201,199:201] = 1
mazemap[49:200,49:51] = 1
mazemap[99:101,99:201] = 1
mazemap = mazemap

sampling_maze = np.zeros((300, 300))
sampling_maze[0:15, :] = 1
sampling_maze[-15:, :] = 1
sampling_maze[:, 0:15] = 1
sampling_maze[:, -15:] = 1
sampling_maze[140:160,40:260] = 1
sampling_maze[190:210,190:310] = 1
sampling_maze[240:260,40:260] = 1
sampling_maze[40:60, 40:260] = 1
sampling_maze[90:210,190:210] = 1
sampling_maze[40:210,40:60] = 1
sampling_maze[90:110,90:210] = 1

# from matplotlib import pyplot as plt
# plt.imshow(mazemap.transpose(), origin='lower')
# plt.show()

def convert_3D_to_2D(xy):
    """
    Convert point_mass.py's x,y to this
    """
    xy = ((xy + 1.5) * 100).astype(np.int)
    return xy

def convert_2D_to_3D(xy):
    xy = (xy / 100.) - 1.5
    return xy

class MotionPlanner:
    def __init__(self, make_new=False):

        dir_path = os.path.dirname(os.path.realpath(__file__))

        # First setup the environment
        map_data = mazemap
        planning_env = MapEnvironment(map_data, sampling_maze)

        if not make_new:
            G = load_graph(os.path.join(dir_path, "graph.pkl"))

        else:
            G = make_graph(planning_env,
                sampler=Sampler(planning_env),
                num_vertices=300,
                connection_radius=50,
                saveto="graph.pkl")
            # input('check')
            # planning_env.visualize_graph(G)

        self.G = G
        self.planning_env = planning_env

        self.plans = dict()

    def state_validity_checker(self, configs_3D, use_sampling_map=True):
        states = convert_3D_to_2D(configs_3D)
        return self.planning_env.state_validity_checker(states, use_sampling_map)

    def motion_plan(self, start, goal):

        start = convert_3D_to_2D(start)
        goal = convert_3D_to_2D(goal)

        if tuple(goal) in self.plans:
            # retreive existing plan
            path = self.plans[tuple(goal)]
            dist = np.linalg.norm(start - path, axis=1)
            idx = min(len(dist) - 1, np.argmin(dist) + 1)
            if dist[idx] < 50 and self.planning_env.edge_validity_checker(start, path[idx]):
                path = np.concatenate([[start], path[idx:]], axis=0)
                return convert_2D_to_3D(path)

        planning_env = self.planning_env

        valid = planning_env.state_validity_checker(np.array([start]))
        if not np.all(valid):
            return False

        # Add start and goal nodes
        G, start_id = add_node(self.G, start, env=planning_env,
            connection_radius=50)
        G, goal_id = add_node(G, goal, env=planning_env,
            connection_radius=50)

        # Uncomment this to visualize the graph
        # planning_env.visualize_graph(G)
        # import sys; sys.exit(0)

        while True:
            try:
                heuristic = lambda n1, n2: planning_env.compute_heuristic(
                    G.nodes[n1]['config'], G.nodes[n2]['config'])

                path = astar_path(G,
                    source=start_id, target=goal_id, heuristic=heuristic)

                # planning_env.visualize_plan(G, path)

                # path = planning_env.shortcut(G, path)
                # planning_env.visualize_plan(G, path, "path.png")
                configs = planning_env.get_path_on_graph(G, path)

                self.plans[tuple(goal)] = configs

                return convert_2D_to_3D(configs)

            except nx.NetworkXNoPath as e:
                # print("failed to plan, make new graph")
                return False
                # G = make_graph(planning_env,
                # sampler=Sampler(planning_env),
                # num_vertices=300,
                # connection_radius=150)

                # # Add start and goal nodes
                # G, start_id = add_node(G, start, env=planning_env,
                #     connection_radius=150)
                # G, goal_id = add_node(G, goal, env=planning_env,
                #     connection_radius=150)
