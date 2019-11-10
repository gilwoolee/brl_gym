import glob
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib; matplotlib.use('PDF')

matplotlib.rc('font', family='serif', size=25)
matplotlib.rc('text', usetex=True)

from matplotlib import pyplot as plt
import brl_gym.scripts.colors as colors

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

    max_step = 5000

    we = [398.391, 1.96*12.22959419954]
    random = 500 * 0.1 + -50 * 0.9
    maximum = 500

    plt.plot([0, max_step], [500, 500], color=colors.max_color, lw=4)
    # plt.text(max_step + 40, maximum - 10, r'Optimal$^*$', color='k')

    # plt.fill_between([0,max_step], y1=[we[0]-we[1],we[0]-we[1]],
    #     y2=[we[0]+we[1],we[0]+we[1]], alpha=0.3, color=colors.expert_color)
    plt.plot([0, max_step], [we[0],we[0]],  color=colors.expert_color, lw=4)
    # plt.text(max_step + 40, we[0] - 10, r'Expert', color='#597223')

    if name == "baseline":
        plt.ylim(0, 500)
        plt.yticks(np.around([0, round(we[0]), round(maximum)]))

        plt.plot([0, max_step], [random, random], color=colors.random_color, lw=4)
        # plt.text(max_step + 40, random - 10, r'Random', color='#878787')
    elif name == "ent_input":
        plt.ylim(0, 500)
        plt.yticks([0, round(we[0]), round(maximum)])
        # plt.plot([0, max_step], [random, random], color=colors.random_color, lw=4)

    else:
        plt.ylim(0, 500)
        plt.yticks([0, round(we[0]), round(maximum)])
        # plt.ylim(0, 500)

    plt.xlim(0, max_step)
    plt.xticks([max_step])
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
            lw=4, color=algo_to_alg[pr][1][colors.EMPHASIS])

        print (algo_to_alg[pr][1][colors.EMPHASIS], algo_to_alg[pr][1][colors.SHADE])
        if timestep < max_step:
            # extend the line
            plt.plot([rewards[-1,0], max_step], [rewards[-1,1], rewards[-1,1]],
                '-', lw=4, color=algo_to_alg[pr][1][colors.EMPHASIS])


    # legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
    #           ncol=3, frameon=False)


    plt.savefig('{}_{}.pdf'.format(env, name), bbox_inches='tight')
    print('{}_{}.pdf'.format(env, name))
    # plt.show()



