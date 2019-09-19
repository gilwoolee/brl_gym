import glob
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib; matplotlib.use('PDF')

matplotlib.rc('font', family='serif', size=15)
matplotlib.rc('text', usetex=True)

from matplotlib import pyplot as plt

algo_to_alg = {
    # "expert_no_residual": ["{BPO-Expert-Input}",'c'],
    # "single_expert_rbpo": "mle-exp-rbpo",
    # "entropy_hidden_rbpo": "rbpo-no-bel-inp",
    "rbpo_stronger_expert": ["{\\bf RBPO}",'r'],
    # "entropy_rbpo": "rbpo-ent",
    "bpo": ["BPO",'g'],
    "upmle": ["UPMLE",'k']
    # "noentropy_rbpo": "rbpo-no-ent-rew",
    # "rbpo_hidden_belief_no_ent_reward": "rbpo-no-ent-rew-no-bel-inp",
    # "rbpo_high_sensing_cost": "",
    # "rbpo": ["{\\bf RBPO}",'r'],
    # "rbpo-ent-10": ["{\\bf RBPO-ent-10}",'g'],
    # "rbpo-ent-100": ["{\\bf RBPO-ent-100}",'b'],
    # "rbpo-ent-10": "",
    # "rbpo-ent-100": "",
    # "noent_rbpo":"",
}

algnames = list(algo_to_alg.keys())
algnames.sort()
# color = ['r','g','b','k','m','c', 'y', ]
stat = dict()

fig, ax = plt.subplots(figsize=(6,4))

env = ["maze", "maze-high-sensing-cost"]
env = env[0]

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

max_step = 6500
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
    plt.plot(rewards[:max_step,0], rewards[:max_step,1], label=algo_to_alg[pr][0], lw=2, color=algo_to_alg[pr][1])


we = [324.95, 17.87]
mle = [271.39, 21.19]
plt.fill_between([0,max_step], y1=[we[0]-we[1],we[0]-we[1]], y2=[we[0]+we[1],we[0]+we[1]], alpha=0.3, color=[0.8,0.8,0])
plt.plot([0, max_step], [we[0],we[0]], label='Expert', color=[0.8,0.8,0])

plt.fill_between([0,max_step], y1=[mle[0]-mle[1],mle[0]-mle[1]], y2=[mle[0]+mle[1],mle[0]+mle[1]], alpha=0.3, color=[0.0,0.0,0.8])
plt.plot([0, max_step], [mle[0],mle[0]], label='MLE-expert', color=[0.0,0.0,0.8])

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45),
          ncol=2, fancybox=True)

# plt.title("Maze")
plt.xlabel("Training Epochs")
plt.ylabel("Total Return")
#plt.savefig('{}_eval_sensing_plot.png'.format(env[1]))
plt.savefig('{}_eval_plot.png'.format(env), bbox_inches='tight')
print('{}_eval_plot.png'.format(env))
# plt.show()



