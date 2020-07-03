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

for name in ["ent", "ent_input", "baseline"]:

    if name == "ent":
        # ent weights
        algo_to_alg = {
            "rbpo_ent100_alpha_0.1": ["100",colors.ent_reward_100],
            "rbpo_ent10_alpha_0.1": ["10",colors.ent_reward_10],
            "rbpo_noent_alpha_0.1": ["0",colors.brpo_color],
            # "rbpo_noent_alpha_1": ["0-alpha1",'g'],

        }
        name = "ent"
        algnames = ['rbpo_ent10_alpha_0.1',
                    'rbpo_ent100_alpha_0.1',
                    'rbpo_noent_alpha_0.1',
                    # 'rbpo_noent_alpha_1'
                    ]
    elif name == "ent_input":
        # ent input
        algo_to_alg = {
            "rbpo_noent_alpha_0.1": ["B+E",colors.brpo_color],
            "rbpo_noent_enthidden_with_expert_alpha_0.1": ["E",colors.ensemble_color],
            # "rbpo_noent_hidden_alpha_0.1": ["None",colors.none_color]
        }
        name = "ent_input"
        algnames = [
        'rbpo_noent_enthidden_with_expert_alpha_0.1',
                    # 'rbpo_noent_hidden_alpha_0.1',
                    'rbpo_noent_alpha_0.1']
    else:
        # baselines
        algo_to_alg = {
            "bpo_ent100": ["BPO",colors.bpo_color],
            "upmle_ent100": ["UPMLE",colors.upmle_color],
            "rbpo_noent_alpha_0.1": [r'\bf{BRPO}',colors.brpo_color]
        }
        name = "baseline"
        algnames = ['bpo_ent100', 'upmle_ent100',
                    'rbpo_noent_alpha_0.1']


    stat = dict()

    fig, ax = plt.subplots(figsize=(8,6))
    env = "maze10easy"

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    max_step = 1250

    we = [398.391, 1.96*12.22959419954]
    maximum = 0
    minimum = 0

    plt.plot([0, max_step], [we[0],we[0]],  color=colors.expert_color, lw=lw)

    for i, pr in enumerate(algnames):
        files = glob.glob("/home/gilwoo/output/{}/{}/*.txt".format(env, pr))
        files.sort()

        if len(files) == 0:
            continue

        rewards = []
        for f in files:

            try:

                data = np.genfromtxt(f, delimiter='\t', skip_header=1)
            except:
                continue

            if data.shape[0] < 5:
                continue
            timestep = int(f.split("/")[-1].split(".")[0])
            if timestep > max_step:
                continue
            rewards += [(timestep, np.mean(data[:, 1]), 1.96*np.std(data[:,1])/np.sqrt(data.shape[0]))]

        rewards = np.array(rewards)
        stat[pr] = rewards
        plt.fill_between(rewards[:,0], y1=rewards[:,1] - rewards[:,2],
            y2=rewards[:max_step,1]+rewards[:max_step,2],
            color=algo_to_alg[pr][1][colors.STANDARD],
            alpha=0.3)
        plt.plot(rewards[:,0], rewards[:,1], label=algo_to_alg[pr][0],
            lw=lw, color=algo_to_alg[pr][1][colors.EMPHASIS])

        if timestep < max_step:
            # extend the line
            plt.plot([rewards[-1,0], max_step], [rewards[-1,1], rewards[-1,1]],
                '-', lw=lw, color=algo_to_alg[pr][1][colors.EMPHASIS])
        maximum = max(maximum, max(rewards[:,1]))
        # minimum = min(minimum, min(rewards[:,1]))

    # legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
    #           ncol=3, frameon=False)

    plt.xticks(ticks=[max_step], labels=[format(max_step, "10.1E")])
    maximum = (maximum +100)// 100 * 100
    yticks = [0, "+"+str(int(maximum))]

    plt.yticks(ticks=[0, maximum], labels=yticks)
    print("max step", max_step)
    plt.xlim(0, max_step)
    plt.ylim(0, maximum )
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    import matplotlib.transforms
    # Create offset transform by 5 points in x direction
    dx = -45/72.; dy = 0/72.
    xoffset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + xoffset)

    dx = 0/72.; dy = -10/72.
    yoffset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.yaxis.get_majorticklabels()[-1:]:
        label.set_transform(label.get_transform() + yoffset)

    dx = 0/72.; dy = +10/72.
    yoffset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.yaxis.get_majorticklabels()[:1]:
        label.set_transform(label.get_transform() + yoffset)

    plt.savefig('{}_{}.pdf'.format(env, name), bbox_inches='tight')
    print('{}_{}.pdf'.format(env, name))
    # plt.show()



