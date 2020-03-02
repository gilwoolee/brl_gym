import glob
import numpy as np

from matplotlib import pyplot as plt
import matplotlib;
matplotlib.use('PDF')
import brl_gym.scripts.colors as colors
import brl_gym.scripts.util as util
lw = util.line_width


matplotlib.rc('font', family='serif', size=25)
matplotlib.rc('text', usetex=True)

for type_name in ["entropy_reward", "ent_input", "baseline"]:

    if type_name == "ent_input":
        # Ent input
        algo_to_alg = {
            "rbpo_noent": ["B+E",colors.brpo_color],
            "entropy_hidden_rbpo": ["E",colors.ensemble_color],
            # "entropy_hidden_no_expert_input_rbpo_noent": ["None",colors.none_color]
        }
        type_name = "ent_input"
        algnames = ["entropy_hidden_rbpo",
         # "entropy_hidden_no_expert_input_rbpo_noent",
          "rbpo_noent", ]
    elif type_name == "entropy_reward":

        # Ent reward
        algo_to_alg = {
            # "rbpo": ["1",'g'],
            "rbpo_noent": ["0",colors.brpo_color],
            "rbpo_ent_10": ["10",colors.ent_reward_10],
            "rbpo_ent_100": ["100",colors.ent_reward_100]
        }
        type_name = "entropy_reward"
        algnames = ["rbpo_noent", "rbpo_ent_10", "rbpo_ent_100"]
        algnames.reverse()

    else:
        # baselines
        algo_to_alg = {
            "bpo": ["BPO",colors.bpo_color],
            "upmle": ["UPMLE",colors.upmle_color],
            "rbpo_noent": [r'\bf{BRPO}',colors.brpo_color]
        }
        type_name = "baseline"
        algnames = ['upmle', 'bpo', 'rbpo_noent']

    stat = dict()


    name = "reward"
    indices = {"reward": 1, "sensing": 3, "crashing": 5}
    index = indices[name]

    max_step = 2500
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    we = [88.84, 2.0830669696387583 * 1.96]
    maximum = 100

    if type_name == 'baseline':
        padding = 3
    else:
        padding = 1
    # we = [78.29, 3.039812] # sensing
    if name == "reward" or name == "sensing":
        plt.plot([0, max_step], [we[0],we[0]],
            color=colors.expert_color, lw=lw)

    plt.xlim(0, max_step)

    for i, pr in enumerate(algnames):
        files = glob.glob("/home/gilwoo/output/doors/{}/*.txt".format(pr))
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
            # print(f)
            # print(data.shape)
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
        stat[pr] = rewards
        plt.fill_between(rewards[:,0],
            y1=rewards[:,index] - 1.96*rewards[:,index+1],
            y2=rewards[:,index]+1.96*rewards[:,index+1],
            alpha=0.3, color=algo_to_alg[pr][1][colors.STANDARD])
        plt.plot(rewards[:,0], rewards[:,index],
            label=algo_to_alg[pr][0], lw=lw,
            color=algo_to_alg[pr][1][colors.EMPHASIS])


    # legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
    #           ncol=3, frameon=False)


    # if type_name == "baseline":
    # plt.ylim([0, 100])
    # elif type_name == "ent_input":
    #     # pass
    #     plt.ylim([we[0]-we[1],100])
    # else:
    #     plt.ylim([we[0]-we[1], 100])
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    plt.savefig('doors_{}.pdf'.format(type_name),
         bbox_inches='tight')
    print('doors_{}.pdf'.format(type_name))
    # plt.show()
# plt.savefig('eval_sensing_plot.png')
# plt.savefig('eval_plot.png')
# plt.show()
# plt.plot(bpo_rewards)
