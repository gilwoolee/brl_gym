import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

#assert(nx.__version__ == '2.2' or nx.__version__ == '2.1')

def load_graph(filename):
    assert os.path.exists(filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['G']

def make_graph(env, sampler, connection_radius, num_vertices, saveto=None):
    """
    Returns a graph ont he passed environment.
    All vertices in the graph must be collision-free.

    Graph should have node attribute "config" which keeps a configuration in tuple.
    E.g., for adding vertex "0" with configuration np.array([0, 1]),
    G.add_node(0, config=tuple(config))

    To add edges to the graph, call
    G.add_weighted_edges_from(edges)
    where edges is a list of tuples (node_i, node_j, weight),
    where weight is the distance between the two nodes.

    @param env: Map Environment for graph to be made on
    @param sampler: Sampler to sample configurations in the environment
    @param connection_radius: Maximum distance to connect vertices
    @param num_vertices: Minimum number of vertices in the graph.
    @param lazy: If true, edges are made without checking collision.
    @param saveto: File to save graph and the configurations
    """

    G = nx.Graph()
    configs = []
    batch_size = 10

    i = 0
    while i < num_vertices:
        sample_configs = sampler.sample(batch_size)
        validity = env.state_validity_checker(sample_configs, use_sampling_map=True)
        for config, valid in zip(sample_configs, validity):
            if not valid or tuple(config) in configs:
                continue
            G.add_node(i, config=tuple(config))
            configs += [tuple(config)]
            i += 1
            if i == num_vertices:
                break

    configs = np.array(configs)

    num_vertices = len(configs)
    total_num_edges = 0
    for i in range(num_vertices - 1):
        distances = env.compute_distances(configs[i], configs[i+1:])
        indices = np.arange(len(distances))[distances <= connection_radius] + i + 1
        edges = [(i, j, distances[j - i - 1]) for j in indices
            if env.edge_validity_checker(configs[i], configs[j], use_sampling_map=True)[0]]

        total_num_edges += len(edges)
        G.add_weighted_edges_from(edges)
    print("Connected {} edges".format(total_num_edges))

    # Save the graph to reuse.
    if saveto is not None:
        data = dict(G=G)
        pickle.dump(data, open(saveto, 'wb'))
        print('Saved the graph to {}'.format(saveto))
    return G


def add_node(G, config, env, connection_radius):
    """
    This function should add a node to an existing graph G.
    @param G graph, constructed using make_graph
    @param config Configuration to add to the graph
    @param env Environment on which the graph is constructed
    @param connection_radius Maximum distance to connect vertices
    @param start_from_config True if config is the starting configuration
    """
    # new index of the configuration
    index = G.number_of_nodes()
    G.add_node(index, config=tuple(config))
    G_configs = nx.get_node_attributes(G, 'config')
    G_configs = [G_configs[node] for node in G_configs]

    # TODO: add edges from the newly added node
    distances = env.compute_distances(G_configs[index], G_configs[:-1])
    indices = np.arange(len(distances))[distances <= connection_radius]

    edges = [(index, j, distances[j]) for j in indices
        if env.edge_validity_checker(G_configs[index], G_configs[j], use_sampling_map=True)[0]]
    G.add_weighted_edges_from(edges)

    return G, index
