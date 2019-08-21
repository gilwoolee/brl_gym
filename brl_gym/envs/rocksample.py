import numpy as np
import os
import json
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle

import gym
from gym.spaces import Discrete
from gym import utils
from gym.utils import seeding


# As described in Smith and Simons (2004): https://arxiv.org/pdf/1207.4166.pdf

class StateType:
    BAD_ROCK = 0
    GOOD_ROCK = 1
    START = 2
    GOAL = 3
    FREE = 4


class Action:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    SAMPLE = 4

class Sense:
    GOOD = 1
    BAD = 0
    NULL = 2


def load_env(filename="rocksample7x4.json", return_dict=False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, "..", "resource", filename)
    if not os.path.exists(filename):
        raise ValueError("{} does not exist".format(filename))
    with open(filename, 'r') as f:
        env_kwargs = json.load(f)

    env = RockSample(**env_kwargs)

    if return_dict:
        return env, env_kwargs
    else:
        return env


class RockSample(gym.Env, utils.EzPickle):

    def __init__(self, grid_size=7, num_rocks=8,
            start_coords=None,
            rock_positions=None,
            good_rock_probability=0.5,
            start_rock_state=None):

        if num_rocks > grid_size ** 2:
            raise ValueError("Number of rocks cannot exceed possible positions")

        self.seed()

        self.grid_size = grid_size
        self.num_rocks = num_rocks
        self.good_rock_probability = good_rock_probability

        # size of action space
        self.nA = self.num_rocks + 5  # [Check, Sample, Move]
        self.nS = (grid_size ** 2) * 3 + 1# position * sensing result

        self.default_start_coords = start_coords
        if start_coords is None:
            start_coords = self.np_random.randint(0, self.grid_size, 2)
        start_coords = np.array(start_coords)

        if np.shape(start_coords)[0] != 2:
            raise ValueError("Start state should be of dimension 2.")

        # Initialize the grid.
        self.grid = np.ones([grid_size, grid_size], dtype='int8') * StateType.FREE
        self.grid[start_coords[0], start_coords[1]] = StateType.START

        if start_rock_state is None:
            start_rock_state = self._random_start_rock_state()
            self.random_start = True
        else:
            assert len(start_rock_state) == num_rocks
            self.random_start = False
        self.start_rock_state = np.array(start_rock_state)

        num_positions = grid_size * grid_size
        if rock_positions is None:
            rock_positions = self.np_random.permutation(num_positions)[:self.num_rocks]
            rock_positions = np.unravel_index(rock_positions, (self.grid_size, self.grid_size))
            rock_positions = np.array(rock_positions).transpose()
        self.rock_positions = tuple(
            [np.array([rock_positions[i][0], rock_positions[i][1]]) for i in range(self.num_rocks)])

        self.start_coords = start_coords
        self.start_state = np.concatenate((start_coords, self.start_rock_state))

        self.action_space = Discrete(self.nA)
        self.observation_space = Discrete(self.nS)

        self.state = None
        self.rock_estimates = None
        self.rock_accuracies = None

        self.horizon = 100
        self.reset()

        # positions are (y, x)
        self.increments = {
            Action.UP: np.array([0, 1], dtype='int8'),
            Action.DOWN: np.array([0, -1], dtype='int8'),
            Action.LEFT: np.array([-1, 0], dtype='int8'),
            Action.RIGHT: np.array([1, 0], dtype='int8'),
        }

    def set_state(self, coords):
        if len(coords) == 2:
            self.state[:2] = coords
        else:
            self.state = coords
        return True

    def get_state(self):
        return self.state

    def set_start_rock_state(self, start_rock_state, start_coords=None):
        self.start_rock_state = start_rock_state
        if not start_coords:
            self.start_state = np.concatenate((self.start_coords,
                start_rock_state))
        else:
            self.start_state = np.concatenate((start_coords,
                start_rock_state))
            self.start_coords = start_coords
        self.random_start = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation(self, coordinates, estimate):
        if coordinates[0] == -1:
            return self.nS - 1
        state = (coordinates[0] * self.grid_size + coordinates[1])
        row = self.grid_size ** 2 * estimate
        state = int(state + row)
        # coords, est =  self._decode_observation(state)
        # assert np.all(np.array(coords) == np.array(coordinates))
        # assert (est == estimate)

        return state

    def _decode_observation(self, obs):
        if isinstance(obs, int) and obs == self.observation_space.n:
            return [-1, -1], Sense.NULL

        row_size = self.grid_size ** 2
        estimate = obs // row_size
        coords = obs % row_size
        x = coords // self.grid_size
        y = coords % self.grid_size

        return [x, y], estimate


    def _decode_observations(self, obs):
        row_size = self.grid_size ** 2
        estimate = obs // row_size
        coords = obs % row_size
        x = coords // self.grid_size
        y = coords % self.grid_size

        return np.array([x, y]).transpose()

    def _random_start_rock_state(self):
        start_rock_state = self.np_random.choice(
            (StateType.GOOD_ROCK, StateType.BAD_ROCK),
            self.num_rocks,
            p=[self.good_rock_probability, 1 - self.good_rock_probability])
        return start_rock_state

    def reset(self):
        if self.random_start:
            self.start_rock_state = self._random_start_rock_state()
            self.start_state = np.concatenate((self.start_state[:2],
                self.start_rock_state))

        self.state = self.start_state
        self.grid[tuple(np.asarray(self.rock_positions).T)] = self.start_rock_state

        self.rock_estimates = np.ones(self.num_rocks) * StateType.GOOD_ROCK
        self.rock_accuracies = np.ones(self.num_rocks) * self.good_rock_probability
        distances = self.get_distances_to_rocks(self.state[:2])

        self.prev_state = self.state.copy()
        self.done = False
        self.timestep = 0

        return self._get_observation(
            self.state[:2], Sense.NULL)

    def reset_state_and_step(self, state, action):
        self.reset()
        self.set_state(state)
        o, r, d, i = self.step(action)
        return o, r, d, i, self.get_state()

    def step(self, action):
        if action >= self.nA:
            raise ValueError("action out of range")
        if (self.state[:2] == -1).all():
            raise ValueError("Cannot continue from terminal state.")

        next_grid = self.grid
        next_coords = self.state[:2]
        next_rock_estimates = self.rock_estimates
        next_rock_accuracies = self.rock_accuracies
        self.prev_state = self.state.copy()
        self.prev_grid = self.grid.copy()
        self.last_action = action

        rock_estimate = Sense.NULL
        sensor_accuracy = 0
        done, reward = False, 0
        sampled_rock_idx = None

        if action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
            direction = self.increments.get(action, np.zeros(2, dtype='int8'))
            next_coords = next_coords + direction
            clip_coords = np.clip(next_coords, 0, self.grid_size-1)

            # Fell off the grid: transition to terminal state
            if (next_coords != clip_coords).any():
                next_coords = -np.ones(2, dtype='int8')
                done = True
                reward = 10 if action == Action.RIGHT else 0 # -100
        elif action == Action.SAMPLE:
            state_type = self.grid[tuple(next_coords)]
            sampled_rock_idx = self.get_rock_idx(next_coords)

            if state_type == StateType.GOOD_ROCK:
                reward = 10

                next_grid = self.grid.copy()
                next_grid[tuple(next_coords)] = StateType.BAD_ROCK
            elif state_type == StateType.BAD_ROCK:
                reward = -10
            else:
                # Sampled with no rock: transition to terminal state
                next_coords = -np.ones(2, dtype='int8')
                done, reward = True, 0 # -100
        else:
            dist = np.linalg.norm(next_coords - self.rock_positions[action-5])
            sensor_efficiency = 2 ** (-dist/20)
            sensor_accuracy = 0.5 + 0.5 * sensor_efficiency

            true_state_type = self.grid[tuple(self.rock_positions[action - 5])]
            if true_state_type == StateType.GOOD_ROCK:
                wrong_state_type = StateType.BAD_ROCK
            else:
                wrong_state_type = StateType.GOOD_ROCK

            rock_estimate = self.np_random.choice(
                (true_state_type, wrong_state_type),
                p=(sensor_accuracy, 1-sensor_accuracy)
            )

            next_rock_accuracies = self.rock_accuracies.copy()
            next_rock_accuracies[action-5] = sensor_accuracy

            next_rock_estimates = self.rock_estimates.copy()
            next_rock_estimates[action-5] = rock_estimate

        self.last_reward = reward
        self.grid = next_grid
        self.state = np.concatenate(
            (next_coords, next_grid[tuple(np.asarray(self.rock_positions).T)]))
        self.rock_estimates = next_rock_estimates
        self.rock_accuracies = next_rock_accuracies
        self.timestep += 1

        self.done = done

        if self.horizon == self.timestep:
            self.done = True
            done = True

        distances = self.get_distances_to_rocks(next_coords)

        return self._get_observation(next_coords, rock_estimate), \
            reward, done, {"sensor_accuracy": sensor_accuracy,
                "coordinates": next_coords, "estimate": rock_estimate,
                "sampled_rock_idx": sampled_rock_idx,
                "coords": next_coords}

    def get_rock_idx(self, coordinates):
        rock_positions = tuple([tuple(i) for i in self.rock_positions])
        if tuple(coordinates) in rock_positions:
            return rock_positions.index(tuple(coordinates))
        else:
            return -1

    def get_distances_to_rocks(self, coordinates):
        distances = [np.linalg.norm(coordinates - i) for i in self.rock_positions]
        return np.array(distances)

    def render(self):
        action_names = ["right" ,"left", "down", "up", "sample"]

        fig, ax = plt.subplots()

        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(0, self.grid_size));
        ax.set_yticks(np.arange(0, self.grid_size));

        grid = np.zeros(self.grid.shape)
        grid[:, -1] = StateType.GOOD_ROCK # Draw Goal with "Green"

        rock_positions = np.array(self.rock_positions)

        prev_state = self.prev_state[:2]
        action = self.last_action
        grid = self.grid
        reward = self.last_reward
        state = self.state

        if action == Action.SAMPLE:
            ax.scatter(
                state[0] + 0.3,
                state[1] + 0.5, c='magenta', s=150)
        if action > Action.SAMPLE:
            rock_idx = action - 5
            ax.scatter(
                rock_positions[rock_idx, 0] + 0.3,
                rock_positions[rock_idx, 1] + 0.5, c='cyan', s=50)

        for i, rp in enumerate(self.rock_positions):
            ax.add_patch(Rectangle((rp[0], rp[1]), 1.0, 1.0, edgecolor='k',facecolor='k', alpha=0.1))

        if self.done:
            if reward == -100:
                ax.text(prev_state[0], prev_state[1], "DEAD", fontsize=15)
            else:
                ax.text(prev_state[0], prev_state[1], "DONE", fontsize=15)
        else:
            ax.plot([prev_state[0] + 0.5, state[0] + 0.5],
                    [prev_state[1] + 0.5, state[1] + 0.5], 'r')
            ax.scatter(state[0] + 0.5, state[1] + 0.5, c='r', s=100)

        plt.xlim(0, self.grid_size)
        plt.ylim(0, self.grid_size)

        if action <= 4:
            plt.title(action_names[action] + " Reward {}".format(reward))
        else:
            plt.title("Sense ({},{}) Reward {}".format(
                self.rock_positions[rock_idx][0],
                self.rock_positions[rock_idx][1],
                reward))

        plt.savefig("path_{}.png".format(self.timestep))
        return fig, ax, plt

    def create_action_path(self, start, rocks_to_sample_in_order, goal=None):
        positions = [start]
        positions += [self.rock_positions[idx] for idx in rocks_to_sample_in_order]

        if goal is None:
            if rocks_to_sample_in_order is not None:
                last_rock_position = self.rock_positions[rocks_to_sample_in_order[-1]]
                goal = last_rock_position.copy()
                goal[1] = self.grid_size + 1
            else:
                goal = list(start)
                goal[1] = self.grid_size + 1

        positions += [goal]

        action_path = []
        coords = positions[0]
        for position in positions[1:]:
            dcol = position[1] - coords[1]
            drow = position[0] - coords[0]

            action = Action.RIGHT if np.sign(dcol) == 1 else Action.LEFT
            action_path += [action] * abs(dcol)
            action = Action.UP if np.sign(drow) == 1 else Action.DOWN
            action_path += [action] * abs(drow)

            if not np.all(position == goal):
                action_path += [Action.SAMPLE]

            coords = position


        return action_path

    def simulate(self, action_path):
        self.reset()
        path = []
        for action in action_path:
            path += [self.step(action)]
            if path[-1][2]:
                return path
        return path

    def compute_discounted_return(self, action_path, gamma=0.95):
        from baselines.common.math_util import discount
        path = self.simulate(action_path)
        rewards = np.array([p[1] for p in path])
        assert not (np.any(rewards < 0))
        return discount(rewards, gamma)[0]
