#!/usr/bin/env python

import argparse, numpy, time
import networkx as nx
import numpy as np

from MapEnvironment import MapEnvironment
import graph_maker
import astar
from Sampler import Sampler
from maze import mazemap

# Try running the following
# python run.py -m ../maps/map1.txt -s 20 20 -g 270 270 --num-vertices 15
# python run.py -m ../maps/map2.txt -s 321 148 -g 106 202 --num-vertices 250 --connection-radius 100

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='script for testing planners')


    parser.add_argument('-m', '--map', type=str, default='map1.txt',
                        help='The environment to plan on')
    parser.add_argument('-s', '--start', nargs='+', type=int, required=True)
    parser.add_argument('-g', '--goal', nargs='+', type=int, required=True)
    parser.add_argument('--num-vertices', type=int, default=300)
    parser.add_argument('--connection-radius', type=float, default=150.0)

    args = parser.parse_args()

    # First setup the environment
    map_data = mazemap
    planning_env = MapEnvironment(map_data)

    # Make a graph
    G = graph_maker.make_graph(planning_env,
        sampler=Sampler(planning_env),
        num_vertices=args.num_vertices,
        connection_radius=args.connection_radius)

    # Add start and goal nodes
    G, start_id = graph_maker.add_node(G, args.start, env=planning_env,
        connection_radius=args.connection_radius)
    G, goal_id = graph_maker.add_node(G, args.goal, env=planning_env,
        connection_radius=args.connection_radius)

    # Uncomment this to visualize the graph
    # planning_env.visualize_graph(G)

    try:
        heuristic = lambda n1, n2: planning_env.compute_heuristic(
            G.nodes[n1]['config'], G.nodes[n2]['config'])

        path = astar.astar_path(G,
            source=start_id, target=goal_id, heuristic=heuristic)

        # planning_env.visualize_plan(G, path)

        waypoints = planning_env.shortcut(G, path)
        # planning_env.visualize_plan(G, waypoints)
        configs = planning_env.get_path_on_graph(G, waypoints)
        planning_env.visualize_waypoints(configs)

    except nx.NetworkXNoPath as e:
        print(e)
