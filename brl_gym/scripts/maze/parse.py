import glob
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib; matplotlib.use('PDF')
import  brl_gym.scripts.colors as colors
import brl_gym.scripts.util as util

matplotlib.rc('font', family='serif', size=25)
matplotlib.rc('text', usetex=True)

from matplotlib import pyplot as plt

for name in ["entropy_reward", "ent_input", "baseline"]:

    if name == "entropy_reward":
        #Entropy rewards
        algo_to_alg = {
            # "rbpo": ["{1}",'k'],
            "rbpo-ent-10": ["{10}",colors.ent_reward_10],
            "rbpo-ent-100": ["{100}",colors.ent_reward_100],
            "rbpo_noent":["{0}",colors.brpo_color],
        }
        name = "entropy_reward"
        algnames = ['rbpo-ent-10', 'rbpo-ent-100', 'rbpo_noent']
    elif name == "ent_input":
        # entropy-inputs
        algo_to_alg = {
            # "entropy_hidden_no_expert_input_rbpo_noent": ["None",colors.none_color],
            "rbpo_hidden_belief_no_ent_reward": ["E",colors.ensemble_color],
            "rbpo_noent": ["B+E",colors.brpo_color],
        }
        name = "ent_input"
        algnames = [
        # 'entropy_hidden_no_expert_input_rbpo_noent',
        'rbpo_hidden_belief_no_ent_reward',
        'rbpo_noent']
    else:

        # baselines
        algo_to_alg = {
            "bpo_ent_1": ["BPO",colors.bpo_color],
            "upmle_ent1": ["UPMLE",colors.upmle_color],
            "rbpo_noent": [r'\bf{BRPO}',colors.brpo_color]
        }
        name = "baseline"
        algnames = ['upmle_ent1', 'bpo_ent_1', 'rbpo_noent']

    stat = dict()

    fig, ax = plt.subplots(figsize=(8,6))

    env = "maze"

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    max_step = 2400

    we = [229.42, 19.79 * 1.96]
    mle = [209.99333333333334, 22.35825375477617 * 1.96]
    # plt.fill_between([0,max_step], y1=[we[0]-we[1],we[0]-we[1]], y2=[we[0]+we[1],we[0]+we[1]],
    #     alpha=0.3, color=expert_color)
    plt.plot([0, max_step], [we[0],we[0]], color=colors.expert_color, lw=util.line_width)
    # plt.text(5200, we[0] - 15, r'Expert', color='#597223')

    if name == "baseline":
        ticks = np.array([0, we[0], 500])
    plt.xlim(0, max_step)

    minimum = maximum = 0
    for i, pr in enumerate(algnames):
        files = glob.glob("/home/gilwoo/output/{}/{}/*.txt".format(env, pr))
        files.sort()

        print(pr, len(files))

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
            rewards += [(timestep, np.mean(data[:, 1]), np.std(data[:,1])/np.sqrt(data.shape[0]))]

        rewards = np.array(rewards)

        stat[pr] = rewards
        plt.fill_between(rewards[:max_step,0], y1=rewards[:max_step,1] - rewards[:max_step,2], y2=rewards[:max_step,1]+rewards[:max_step,2],
            alpha=0.3, color=algo_to_alg[pr][1][colors.STANDARD])
        plt.plot(rewards[:max_step,0], rewards[:max_step,1],
            label=algo_to_alg[pr][0], lw=util.line_width,
            color=algo_to_alg[pr][1][colors.EMPHASIS])

        maximum = max(maximum, max(rewards[:max_step,1]))
        minimum = min(minimum, min(rewards[:max_step,1]))

    # plt.fill_between([0,max_step], y1=[mle[0]-mle[1],mle[0]-mle[1]], y2=[mle[0]+mle[1],mle[0]+mle[1]], alpha=0.3, color=[0.0,0.0,0.8])
    # plt.plot([0, max_step], [mle[0],mle[0]], label='MLE-expert', color=[0.0,0.0,0.8])


    plt.xticks(ticks=[max_step], labels=[format(max_step*50, "10.1E")])

    plt.yticks(ticks=[0, 500], labels=[0, "+500"])
    plt.ylim(0, 500)

    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    import matplotlib.transforms
    # Create offset transform by 5 points in x direction
    dx = -45/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
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

    plt.savefig('{}_{}.pdf'.format(env, name), bbox_inches='tight')
    print('{}_{}.pdf'.format(env, name))



