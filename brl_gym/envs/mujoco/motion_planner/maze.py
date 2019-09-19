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
from PIL import Image
from .util import convert_3D_to_2D, convert_2D_to_3D

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

class MotionPlanner:
    def __init__(self, maze_type=4, make_new=False):

        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.maze_type = maze_type

        # First setup the environment
        if maze_type == 4:
            map_data = mazemap
            planning_env = MapEnvironment(map_data, sampling_maze, maze_type=maze_type)
            num_vertices = 300
            self.connection_radius = 75
        else:
            mapfile = os.path.join(dir_path, "../assets/walls.png")
            sampling_mapfile = os.path.join(dir_path, "../assets/walls_padding.png")
            img = Image.open(mapfile).convert('L')
            map_data = np.array(img)

            sampling_mapfile = os.path.join(dir_path, "../assets/walls_padding.png")
            sampling_img = Image.open(sampling_mapfile).convert('L')
            sampling_map_data = np.array(sampling_img)

            num_vertices = 500
            self.connection_radius = 25
            planning_env = MapEnvironment(map_data, sampling_map_data, maze_type=maze_type)

        if not make_new:
            if maze_type == 4:
                G = load_graph(os.path.join(dir_path, "resource", "graph.pkl"))
            else:
                G = load_graph(os.path.join(dir_path, "resource","graph10.pkl"))

        else:
            if maze_type == 4:
                G = make_graph(planning_env,
                    sampler=Sampler(planning_env),
                    num_vertices=num_vertices,
                    connection_radius=self.connection_radius,
                    saveto="resource/graph.pkl")
            else:
                G = make_graph(planning_env,
                    sampler=Sampler(planning_env),
                    num_vertices=num_vertices,
                    connection_radius=self.connection_radius,
                    saveto="resource/graph10.pkl")
        #     input('check')
        # planning_env.visualize_graph(G)
        # import sys; sys.exit(0)

        self.G = G
        self.planning_env = planning_env

        self.plans = dict()

    def state_validity_checker(self, configs_3D, use_sampling_map=True):
        states = convert_3D_to_2D(configs_3D, self.maze_type)
        # print('maze', states, configs_3D, self.maze_type)
        return self.planning_env.state_validity_checker(states, use_sampling_map)

    def motion_plan(self, start, goal, reuse=True, shortcut=False):
        start2D = start.copy()
        start = convert_3D_to_2D(start, self.maze_type)
        goal = convert_3D_to_2D(goal, self.maze_type)
        # print("MP", start, start2D, self.maze_type)

        if tuple(goal) in self.plans and reuse:
            # retreive existing plan
            path = self.plans[tuple(goal)]
            dist = np.linalg.norm(start - path, axis=1)
            idx = min(len(dist) - 1, np.argmin(dist) + 1)
            if dist[idx] < 5 and self.planning_env.edge_validity_checker(start, path[idx]):
                path = np.concatenate([[start], path[idx:]], axis=0)
                return convert_2D_to_3D(path, self.maze_type)

        planning_env = self.planning_env

        valid = planning_env.state_validity_checker(np.array([start]))
        if not np.all(valid):
            print(start, start2D)
            print("start not valid")
            return False

        # Add start and goal nodes
        G, start_id = add_node(self.G, start, env=planning_env,
            connection_radius=self.connection_radius)
        G, goal_id = add_node(G, goal, env=planning_env,
            connection_radius=self.connection_radius)

        # Uncomment this to visualize the graph
        # planning_env.visualize_graph(G)
        # print("visualize")
        # import sys; sys.exit(0)

        while True:
            try:
                heuristic = lambda n1, n2: planning_env.compute_heuristic(
                    G.nodes[n1]['config'], G.nodes[n2]['config'])

                path = astar_path(G,
                    source=start_id, target=goal_id, heuristic=heuristic)

                # planning_env.visualize_plan(G, path)

                if shortcut:
                    path = planning_env.shortcut(G, path)
                # planning_env.visualize_plan(G, path)

                configs = planning_env.get_path_on_graph(G, path)

                self.plans[tuple(goal)] = configs

                G.remove_node(start_id)
                G.remove_node(goal_id)
                return convert_2D_to_3D(configs, self.maze_type)

            except nx.NetworkXNoPath as e:

                G.remove_node(start_id)
                G.remove_node(goal_id)

                return False


if __name__ == "__main__":

    mp = MotionPlanner(make_new=True, maze_type=10)
    # mp = MotionPlanner(make_new=False, maze_type=10)