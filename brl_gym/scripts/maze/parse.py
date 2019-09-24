import glob
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib; matplotlib.use('PDF')
from brl_gym.scripts.colors import random_color, max_color, expert_color

matplotlib.rc('font', family='serif', size=25)
matplotlib.rc('text', usetex=True)

from matplotlib import pyplot as plt

for name in ["entropy_reward", "ent_input", "baseline"]:

    if name == "entropy_reward":
        #Entropy rewards
        algo_to_alg = {
            # "rbpo": ["{1}",'k'],
            "rbpo-ent-10": ["{10}","#4B7A4F"],
            "rbpo-ent-100": ["{100}","#795245"],
            "rbpo_noent":["{0}",'#e41a1c'],
        }
        name = "entropy_reward"
        algnames = ['rbpo-ent-10', 'rbpo-ent-100', 'rbpo_noent']
    elif name == "ent_input":
        # entropy-inputs
        algo_to_alg = {
            "entropy_hidden_no_expert_input_rbpo_noent": ["None",'#8C7F70'],
            "rbpo_hidden_belief_no_ent_reward": ["E",'#504E75'],
            "rbpo_noent": ["B+E",'#e41a1c'],
        }
        name = "ent_input"
        algnames = ['entropy_hidden_no_expert_input_rbpo_noent', 'rbpo_hidden_belief_no_ent_reward', 'rbpo_noent']
    else:

        # baselines
        algo_to_alg = {
            "bpo_ent_1": ["BPO",'#577AA3'],
            "upmle_ent1": ["UPMLE",'#9C72A3'],
            "rbpo_noent": [r'\bf{BRPO}','#e41a1c']
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

    max_step = 5000

    we = [229.42, 19.79 * 1.96]
    mle = [209.99333333333334, 22.35825375477617 * 1.96]
    # plt.fill_between([0,max_step], y1=[we[0]-we[1],we[0]-we[1]], y2=[we[0]+we[1],we[0]+we[1]],
    #     alpha=0.3, color=expert_color)
    plt.plot([0, max_step], [we[0],we[0]], color=expert_color, lw=4)
    # plt.text(5200, we[0] - 15, r'Expert', color='#597223')

    maximum = 500
    random = 500 * 0.25 + -500 * 0.75
    plt.plot([0, max_step], [maximum, maximum], color=max_color, lw=4)
    # plt.text(5200, maximum - 10, r'Optimal$^*$', color='k')

    if name == "baseline":
        plt.plot([0, max_step], [random, random], color=random_color, lw=4)
        # plt.text(5200, random - 10, r'Random', color='#878787')
        ticks = np.array([random, 0, we[0], 500])
        plt.yticks(np.around(ticks, 0))

    else:
        plt.yticks([0, round(we[0]), 500])
        plt.ylim(0,500)
    plt.xlim(0, max_step)

    plt.xticks([max_step])

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
            print(f)
            print(data.shape)
            if data.shape[0] < 5:
                continue
            timestep = int(f.split("/")[-1].split(".")[0])
            if timestep > max_step:
                continue
            rewards += [(timestep, np.mean(data[:, 1]), np.std(data[:,1])/np.sqrt(data.shape[0]))]

        rewards = np.array(rewards)
        print(rewards[:5])
        stat[pr] = rewards
        plt.fill_between(rewards[:max_step,0], y1=rewards[:max_step,1] - rewards[:max_step,2], y2=rewards[:max_step,1]+rewards[:max_step,2],
            alpha=0.3, color=algo_to_alg[pr][1])
        plt.plot(rewards[:max_step,0], rewards[:max_step,1], label=algo_to_alg[pr][0], lw=4, color=algo_to_alg[pr][1])



    # plt.fill_between([0,max_step], y1=[mle[0]-mle[1],mle[0]-mle[1]], y2=[mle[0]+mle[1],mle[0]+mle[1]], alpha=0.3, color=[0.0,0.0,0.8])
    # plt.plot([0, max_step], [mle[0],mle[0]], label='MLE-expert', color=[0.0,0.0,0.8])




    # legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
    #           ncol=3, frameon=False)


    plt.savefig('{}_{}.pdf'.format(env, name), bbox_inches='tight')
    print('{}_{}.pdf'.format(env, name))



