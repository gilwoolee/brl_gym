import glob
import numpy as np

from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('PDF')

matplotlib.rc('font', family='serif', size=15)
matplotlib.rc('text', usetex=True)
algo_to_alg = {
    "bpo": ["BPO",'g'],
    "upmle": ["UPMLE", "k"],
    "expert_no_residual": ["BPO-Expert", "c"],
    "rbpo": ["{\\bf RBPO}",'r'],
    # "entropy_hidden_rbpo": ["{\\bf RBPO-NoBelief}", 'c'],
    # "entropy_rbpo": ["{\\bf RBPO-Entropy}", 'b'],
}

algnames = list(algo_to_alg.keys())
algnames.sort()
# color = ['r','g','b','k','m','c', 'y', ]
stat = dict()

name = "reward"
indices = {"reward": 1, "sensing": 3, "crashing": 5}
index = indices[name]

max_step = 5000
fig, ax = plt.subplots(figsize=(6,4))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')


for i, pr in enumerate(algnames):
    files = glob.glob("output/{}/*.txt".format(pr))
    files.sort()

    print(pr, len(files))

    if len(files) == 0:
        continue

    rewards = []
    for f in files:
        data = np.genfromtxt(f, delimiter='\t', skip_header=1)
        print(f)
        print(data.shape)
        if data.shape[0] < 5:
            continue
        timestep = int(f.split("/")[-1].split(".")[0])
        if timestep > max_step:
            continue
        rewards += [(timestep,
            np.mean(data[:, 0]), np.std(data[:,0])/np.sqrt(data.shape[0]),
            np.mean(data[:, 1]), np.std(data[:,1])/np.sqrt(data.shape[0]),
            np.mean(data[:, 2]), np.std(data[:,2])/np.sqrt(data.shape[0]))]

    rewards = np.array(rewards)
    # print(rewards[:5])
    stat[pr] = rewards
    plt.fill_between(rewards[:,0], y1=rewards[:,index] - rewards[:,index+1], y2=rewards[:,index]+rewards[:,index+1],
        alpha=0.3, color=algo_to_alg[pr][1])
    plt.plot(rewards[:,0], rewards[:,index], label=algo_to_alg[pr][0], lw=2, color=algo_to_alg[pr][1])


we = [88.84, 2.0830669696387583]
# we = [78.29, 3.039812] # sensing

plt.fill_between([0,max_step], y1=[we[0]-we[1],we[0]-we[1]], y2=[we[0]+we[1],we[0]+we[1]], alpha=0.3, color=[0.8,0.8,0])
plt.plot([0, max_step], [we[0],we[0]], label='Expert', color=[0.8,0.8,0])

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
          ncol=2, fancybox=True)
plt.savefig('doors_eval_reward_plot.png',
     bbox_inches='tight')
# plt.savefig('eval_sensing_plot.png')
# plt.savefig('eval_plot.png')
# plt.show()
# plt.plot(bpo_rewards)



