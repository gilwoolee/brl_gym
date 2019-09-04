import glob
import numpy as np
from matplotlib import pyplot as plt

prefix = ["bpo", "expert"]
names = ["bpo", "r-bpo"]
stat = dict()

fig = plt.figure()
for i, pr in enumerate(prefix):

    files = glob.glob("{}_*.txt".format(pr))
    files.sort()

    rewards = []
    for f in files:
        data = np.genfromtxt(f, delimiter='\t', skip_header=1)
        timestep = int(f.split("_")[1].split(".")[0])
        rewards += [(timestep,
            np.mean(data[:, 0]), np.std(data[:,0])/np.sqrt(data.shape[0]),
            np.mean(data[:, 1]), np.std(data[:,1])/np.sqrt(data.shape[0]),
            np.mean(data[:, 2]), np.std(data[:,2])/np.sqrt(data.shape[0]))]

    rewards = np.array(rewards)

    # import IPython; IPython.embed(); import sys; sys.exit()
    stat[pr] = rewards
    plt.plot(rewards[:,0], rewards[:,3], label=names[i])
# -108.5296 16.521
# plt.plot([0, 3000], [-108.5, -108.5], label='early-sys-id')

plt.title("sensing")
plt.legend()
plt.show()
# plt.plot(bpo_rewards)


