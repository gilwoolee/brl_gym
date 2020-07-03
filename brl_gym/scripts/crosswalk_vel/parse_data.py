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
plt.ylim(-400, 100)

#plt.xlim(0, max_step)
for i, pr in enumerate(algnames):
    files = glob.glob("data/{}/{}*.txt".format(pr, pr))
    files.sort()

    rewards =[]
    for f in files:
        print(f)
        data = np.genfromtxt(f, delimiter='\t', skip_header=1)
        rewards += [(np.mean(data[:,0]), 1.96*np.std(data[:,0])/np.sqrt(len(data)))]

    rewards = np.array(rewards)
    plt.fill_between(np.arange(len(rewards)), y1=rewards[:,0] - rewards[:,1],
        y2=rewards[:,0]+rewards[:,1],
        color=algo_to_alg[pr][1][colors.STANDARD],
        alpha=0.3)
    plt.plot(np.arange(len(rewards)), rewards[:,0], label=algo_to_alg[pr][0],
        lw=lw, color=algo_to_alg[pr][1][colors.EMPHASIS])

plt.xlim(0, 50)
plt.xticks(ticks=[50], labels=[format(50*50, "10.1E")]) # Each file is 50 epoch apart
plt.yticks(ticks=[-200,100], labels=["-200", "+100"])
plt.ylim(-200, 100)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False)

import matplotlib.transforms
# Create offset transform by 5 points in x direction
dx = -45/72.; dy = 0/72.
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
print(offset)
# apply offset transform to all x ticklabels.
for label in ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

dx = 0/72.; dy = -10/72.
yoffset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
for label in ax.yaxis.get_majorticklabels()[-1:]:
    label.set_transform(label.get_transform() + yoffset)

dx = 0/72.; dy = +10/72.
yoffset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
for label in ax.yaxis.get_majorticklabels()[:1]:
    label.set_transform(label.get_transform() + yoffset)
plt.savefig('{}_reward.pdf'.format(env), bbox_inches='tight')




