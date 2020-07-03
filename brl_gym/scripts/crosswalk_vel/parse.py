import glob
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib; matplotlib.use('PDF')

matplotlib.rc('font', family='serif', size=25)
matplotlib.rc('text', usetex=True)

from matplotlib import pyplot as plt
import brl_gym.scripts.colors as colors
import brl_gym.scripts.util as util

lw = util.line_width

# baselines
algo_to_alg = {
    "bpo": ["BPO",colors.bpo_color],
    "mle": ["UPMLE",colors.upmle_color],
    "brpo": [r'\bf{BRPO}',colors.brpo_color]
}
name = "baseline"
algnames = ['bpo', 'mle', 'brpo']


stat = dict()

fig, ax = plt.subplots(figsize=(8,6))
env = "crosswalk"

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

max_step = 5000

we = [-177.53, 1.96*12.3] # Expert

plt.plot([0, max_step], [we[0],we[0]],  color=colors.expert_color, lw=lw)
#plt.ylim(0, 500)

#plt.xlim(0, max_step)
for i, pr in enumerate(algnames):
    files = glob.glob("data/{}.csv".format(pr))
    files.sort()

    data = np.genfromtxt(files[0], delimiter=',', skip_header=1)
    rewards = np.vstack([data[:, -2], data[:, -1]]).transpose()

    # plt.fill_between(rewards[:,0], y1=rewards[:,1] - rewards[:,2],
    #     y2=rewards[:max_step,1]+rewards[:max_step,2],
    #     color=algo_to_alg[pr][1][colors.STANDARD],
    #     alpha=0.3)
    plt.plot(rewards[:,0], rewards[:,1], label=algo_to_alg[pr][0],
        lw=lw, color=algo_to_alg[pr][1][colors.EMPHASIS])

plt.xlim(0, 50)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

plt.savefig('{}.pdf'.format(env), bbox_inches='tight')




