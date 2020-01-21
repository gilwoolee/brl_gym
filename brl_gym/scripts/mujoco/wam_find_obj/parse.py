import glob
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib; matplotlib.use('PDF')

matplotlib.rc('font', family='serif', size=25)
matplotlib.rc('text', usetex=True)

from matplotlib import pyplot as plt
import brl_gym.scripts.colors as colors

# baselines
algo_to_alg = {
    "bpo": ["BPO",colors.bpo_color],
    # "upmle": ["UPMLE",colors.upmle_color],
    "brpo": [r'\bf{BRPO}',colors.brpo_color]
}
name = "baseline"
algnames = ['bpo', #'upmle_ent100',
            'brpo']


stat = dict()

fig, ax = plt.subplots(figsize=(8,6))
env = "WamFindObj"

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

max_step = 50

# we = [398.391, 1.96*12.22959419954]
random = 500 * 0.1 + -50 * 0.9
maximum = 50

# plt.plot([0, max_step], [500, 500], color=colors.max_color, lw=4)

# plt.plot([0, max_step], [we[0],we[0]],  color=colors.expert_color, lw=4)

#plt.ylim(0, 500)


#plt.xlim(0, max_step)
plt.xticks([max_step])
for i, pr in enumerate(algnames):
    files = glob.glob("data/{}.csv".format(pr))
    files.sort()

    data = np.genfromtxt(files[0], delimiter=',', skip_header=1)
    # import IPython; IPython.embed()
    rewards = np.vstack([data[:, -2], data[:, -1]]).transpose()

    # plt.fill_between(rewards[:,0], y1=rewards[:,1] - rewards[:,2],
    #     y2=rewards[:max_step,1]+rewards[:max_step,2],
    #     color=algo_to_alg[pr][1][colors.STANDARD],
    #     alpha=0.3)
    print(rewards[:,1])
    plt.plot(rewards[:,0], rewards[:,1], label=algo_to_alg[pr][0],
        lw=4, color=algo_to_alg[pr][1][colors.EMPHASIS])


# legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
#           ncol=3, frameon=False)

#plt.xlim(0, 100)

plt.savefig('{}.pdf'.format(env), bbox_inches='tight')

# plt.show()



