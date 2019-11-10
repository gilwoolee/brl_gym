import glob
import numpy as np
import os
from matplotlib import pyplot as plt

# image_files = glob.glob("../../../images/*.jpg")


def convert_3D_to_2D(xy):
    """
    Convert point_mass.py's x,y to this
    """
    xy = xy.copy()
    xy[1] *= -1
    print(xy,)
    xy = ((xy + 1.5) / 3.0 * 525).astype(np.int)
    print(xy)
    return xy


csv_file = "../../../expert_action.csv"
agent_pos = np.genfromtxt("../../../agent_pos.csv", delimiter=",")
expert_actions = np.genfromtxt(csv_file, delimiter=",")[:, :2]
expert_actions = expert_actions / np.linalg.norm(expert_actions, axis=1).reshape(-1,1)
expert_actions *= 100

actions = np.genfromtxt("../../../actions.csv", delimiter=",")[:, :2]
actions = actions / np.linalg.norm(actions, axis=1).reshape(-1,1)
actions *= 100

print(actions.shape, expert_actions.shape)

ax = plt.axes()

# origin = [292, 298]
origin = np.array([30, 44])

for idx, i in enumerate(range(0, 501, 5)):
    file = "../../../images/{}.npy".format(i)

    if not os.path.exists(file):
        continue

    data = np.load(file)
    data = data[365:-70, 125:-310, :]
    expert_action = expert_actions[i].copy()
    expert_action[1] *= -1

    action = actions[i].copy()
    action[1] *= -1

    agent = convert_3D_to_2D(agent_pos[i]) + origin

    plt.imshow(data)

    # ax.arrow(0, 0, action[0], action[1], head_width=50, head_length=10, fc='red', ec='red')
    print(agent, action, expert_action)
    plt.arrow(agent[0], agent[1], expert_action[0], expert_action[1], head_width=15, head_length=15,
        fc='b', ec='b', linestyle=':')
    plt.arrow(agent[0], agent[1], action[0], action[1], head_width=15, head_length=15, fc='r', ec='r')

    plt.axis('off')
    plt.savefig("test{}.png".format(i), bbox_inches='tight')
    plt.clf()


