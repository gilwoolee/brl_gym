import glob
import numpy as np
from matplotlib import pyplot as plt

prefix = ["bpo", "expert"]
stat = dict()

fig = plt.figure()
for pr in prefix:

    files = glob.glob("{}_*.txt".format(pr))
    files.sort()

    rewards = []
    for f in files:
        data = np.genfromtxt(f, delimiter='\t', skip_header=1)
        timestep = int(f.split("_")[1].split(".")[0])
        rewards += [(timestep, np.mean(data[:, 1]), np.std(data[:,1])/np.sqrt(data.shape[0]))]

    rewards = np.array(rewards)

    # import IPython; IPython.embed(); import sys; sys.exit()
    stat[pr] = rewards
    plt.plot(rewards[:,0], rewards[:,1], label=pr)

plt.plot([0, 12000], [229.978, 229.978], label='early-sys-id')

plt.legend()
plt.show()
# plt.plot(bpo_rewards)


