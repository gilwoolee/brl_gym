import numpy as np
from scipy.stats import norm

from gym.spaces import Box
from brl_gym.estimators.estimator import Estimator
from brl_gym.envs.mujoco.point_mass import PointMassEnv
from brl_gym.envs.mujoco.point_mass_slow import PointMassSlowEnv
from brl_gym.envs.mujoco.maze10 import Maze10
from brl_gym.estimators.learnable_bf import LearnableBF

class BayesMazeEstimator(Estimator):
    """
    This class estimates tiger location given a known observation error
    """
    def __init__(self, maze_type=4):
        self.maze_type = maze_type
        if maze_type == 4:
            from brl_gym.envs.mujoco.point_mass import GOAL_POSE
            env = PointMassEnv()
        elif maze_type == (4, 'slow'):
            from brl_gym.envs.mujoco.point_mass import GOAL_POSE
            env = PointMassSlowEnv()
        else:
            from brl_gym.envs.mujoco.maze10 import GOAL_POSE
            env = Maze10()

        self.GOAL_POSE = GOAL_POSE.copy()
        self.belief_high = np.ones(GOAL_POSE.shape[0])
        self.belief_low = np.zeros(GOAL_POSE.shape[0])
        belief_space = Box(self.belief_low, self.belief_high, dtype=np.float32)
        super(BayesMazeEstimator, self).__init__(
                env.observation_space, env.action_space, belief_space)
        # self.reset()
        self.param_low = np.zeros(GOAL_POSE.shape[0])
        self.param_high = np.ones(GOAL_POSE.shape[0])
        self.param_space = Box(self.param_low, self.param_high)

    def reset(self):
        self.belief = np.ones(self.GOAL_POSE.shape[0])
        self.belief /= np.sum(self.belief)
        return self.belief.copy()

    def estimate(self, action, observation, **kwargs):
        """
        Given the latest (action, observation), update belief
        observation: State.CLOSED_LEFT, State.CLOSED_RIGHT
        """
        if action is None:
            return self.reset()

        # obs = observation[3:5]
        if 'goal_dist' in kwargs:
            obs_goal_dist = kwargs['goal_dist']
            noise_scale = kwargs['noise_scale']
            dist_to_goals = np.linalg.norm(observation[4:4+self.GOAL_POSE.shape[0]*2].reshape(-1,2), axis=1)
            p_obs_given_prior = norm.pdf(dist_to_goals - obs_goal_dist, scale=noise_scale) * self.belief
            p_goal_obs  = p_obs_given_prior / np.sum(p_obs_given_prior)
            self.belief = p_goal_obs
            return self.belief
        else:
            return self.belief

    def get_belief(self):
        return self.belief.copy()

    def get_mle(self):
        return np.around(self.belief)

class LearnableMazeBF(LearnableBF, BayesMazeEstimator):
    def __init__(self, maze_type=4):
        BayesMazeEstimator.__init__(self, maze_type)
        LearnableBF.__init__(self, self._action_space, self._observation_space, self.belief_space)

    def reset(self):
        return LearnableBF.reset(self)

    def estimate(self, action, observation, **kwargs):
        return LearnableBF.estimate(self, action, observation, **kwargs)


def main_BayesMazeEstimator():
    maze_type = 10

    if maze_type == 4:
        env = PointMassEnv()
    else:
        env = Maze10()
    estimator = BayesMazeEstimator(maze_type)
    state = env.reset()

    for _ in range(500):
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)

        belief = estimator.estimate(a, obs, **info)
        print ("belief:", np.around(estimator.get_belief(),1))


def main_LearnableMazeBF():
    from brl_gym.envs.mujoco.maze10easy import Maze10Easy
    from brl_gym.estimators.learnable_bf import BayesFilterDataset
    import brl_gym.estimators.learnable_bf.pt_util as pt_util
    import brl_gym.estimators.learnable_bf.util as util
    from brl_gym.estimators.learnable_bf.learnable_bf import generate_labels

    import torch
    import torch.optim as optim
    import tqdm

    # Collect data
    env = Maze10Easy(reset_param=True)
    estimator = LearnableMazeBF(maze_type=10)

    labels = []
    inputs = []
    T = 120
    for _ in tqdm.tqdm(range(100)):
        o = env.reset()
        estimator.reset()

        a = np.zeros(env.action_space.shape[0])
        inp = [np.concatenate([a, o])]

        for _ in range(T - 1):
            a = env.action_space.sample()
            o, r, d, info = env.step(a)
            b = estimator.estimate(a, o, **info)
            # b, output, hidden = estimator(a, o, **info)
            inp += [np.concatenate([a, o])]

        labels += [np.array([env.target] * T, dtype=np.int16)]
        inputs += [np.vstack(inp)]

    inputs = np.stack(inputs)
    labels = np.stack(labels)

    # Setup training / testing
    output_dim = 10
    SEQUENCE_LENGTH = 10
    BATCH_SIZE = 10
    FEATURE_SIZE = 15
    TEST_BATCH_SIZE = 5
    EPOCHS = 50
    LEARNING_RATE = 0.002
    WEIGHT_DECAY = 0.0005
    USE_CUDA = True
    PRINT_INTERVAL = 10
    LOG_PATH = 'logs/log.pkl'

    train_data = inputs[:int(len(inputs)*0.8)]
    train_label = labels[:int(len(labels)*0.8)]

    test_data = inputs[int(len(inputs)*0.8):]
    test_label = labels[int(len(labels)*0.8):]

    data_train = BayesFilterDataset(train_data, train_label, output_dim, SEQUENCE_LENGTH, BATCH_SIZE)
    data_test = BayesFilterDataset(test_data, test_label, output_dim, SEQUENCE_LENGTH, BATCH_SIZE)

    print("train has ", len(data_train))
    use_cuda = USE_CUDA and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    import multiprocessing
    num_workers = multiprocessing.cpu_count()
    print('num workers:', num_workers)

    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE,
                                               shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE,
                                              shuffle=False, **kwargs)

    model = estimator.model.to(device)

    # Adam is an optimizer like SGD but a bit fancier. It tends to work faster and better than SGD.
    # We will talk more about different optimization methods in class.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    start_epoch = model.load_last_model('checkpoints')

    train_losses, test_losses, test_accuracies = pt_util.read_log(LOG_PATH, ([], [], []))
    test_loss, test_accuracy = util.test(model, device, test_loader)

    test_losses.append((start_epoch, test_loss))
    test_accuracies.append((start_epoch, test_accuracy))

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            lr = LEARNING_RATE * np.power(0.25, (int(epoch / 6)))
            train_loss = util.train(model, device, optimizer, train_loader, lr, epoch, PRINT_INTERVAL)
            test_loss, test_accuracy = util.test(model, device, test_loader)
            train_losses.append((epoch, train_loss))
            test_losses.append((epoch, test_loss))
            test_accuracies.append((epoch, test_accuracy))
            pt_util.write_log(LOG_PATH, (train_losses, test_losses, test_accuracies))
            model.save_best_model(test_accuracy, 'checkpoints/%03d.pt' % epoch)
            seed_sequence, seed_label = data_train[0]
            print("seed", seed_label[0])
            generated_labels = generate_labels(model, device, seed_sequence, 'max')
            print('generated max\t\t', generated_labels)
            for ii in range(10):
                generated_labels = generate_labels(model, device, seed_sequence, 'sample')
                print('generated sample\t', generated_labels)
            # print('')

    except KeyboardInterrupt as ke:
        print('Interrupted')
    except:
        import traceback
        traceback.print_exc()
    finally:
        print('Saving final model')
        model.save_model('checkpoints/%03d.pt' % epoch, 0)
        ep, val = zip(*train_losses)
        pt_util.plot(ep, val, 'Train loss', 'Epoch', 'Error')
        ep, val = zip(*test_losses)
        pt_util.plot(ep, val, 'Test loss', 'Epoch', 'Error')
        ep, val = zip(*test_accuracies)
        pt_util.plot(ep, val, 'Test accuracy', 'Epoch', 'Error')

        # Plot perplexity
        ep, val = zip(*train_losses)
        val = np.exp(val)
        pt_util.plot(ep, val, 'Train perplexity', 'Epoch', 'Error')
        ep, val = zip(*test_losses)
        val = np.exp(val)
        pt_util.plot(ep, val, 'Test perplexity', 'Epoch', 'Error')
        print("Final test perplexity was ", val[-1])

        return model, device

    import IPython; IPython.embed(); import sys; sys.exit(0)




# main_LearnableMazeBF()
