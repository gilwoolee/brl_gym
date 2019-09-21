import glob
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib; matplotlib.use('PDF')
from matplotlib import pyplot as plt

matplotlib.rc('font', family='serif', size=30)
matplotlib.rc('text', usetex=True)


# Plot bar chart for Ablation study
entropy = {
    "rbpo_noent":[r'{\bf $\mathbf{\epsilon=0}$}','r'],
    "rbpo-ent-10": [r"$\epsilon=10$",[0.3,0.3,0.3]],
    "rbpo-ent-100": [r"$\epsilon=100$",[0.3,0.3,0.3]],
    "rbpo":[r'$\epsilon=1$',[0.3,0.3,0.3]]
}

alpha = {
    "rbpo_ent10-alpha_0.1": [r'{${0.1}$}'],
    "rbpo_ent10_alpha_0.25": [r'$0.25$'],
    "rbpo_ent10_alpha_0.5": [r'$0.5$'],
    "rbpo_ent10_alpha_1": [r'${1}$']
}

entropy_order = [
    "rbpo_noent",
    "rbpo",
    "rbpo-ent-10",
    "rbpo-ent-100",
    ]

alpha_order = [
    "rbpo_ent10-alpha_0.1",
    "rbpo_ent10_alpha_0.25",
    "rbpo_ent10_alpha_0.5",
    "rbpo_ent10_alpha_1"
]

env_names = {"maze10": r'\texttt{Maze10}'}

optimal = {"maze10": 500}
expert =  {"maze10": [80.73, 23.68*1.96]}
min_random = {"maze10": 0}
rewards_idx = {"maze10":1}
fig, ax = plt.subplots(figsize=(8,6))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

rewards = dict()

alg_type = "alpha"
if alg_type == "entropy":
    algo_names, algo_order = entropy, entropy_order
else:
    algo_names, algo_order = alpha, alpha_order

for i, env in enumerate(env_names):
    idx = rewards_idx[env]
    rewards[env] = []
    names = []
    yerr = []
    for algo in algo_order:
        files = glob.glob("/home/gilwoo/output/{}/{}/*.txt".format(env, algo))
        files.sort()
        if len(files) == 0:
            print(env, algo, " doesn't have any output")
            continue

        reward = []
        for file in files:
            try:
                data = np.genfromtxt(file, delimiter='\t', skip_header=1)
                reward += [[np.mean(data[:, idx]), np.std(data[:,idx])/np.sqrt(data.shape[0])]]
            except:
                print("Failed to parse", file)
                continue

        # Get best performing
        reward = np.array(reward)
        print(reward.shape)
        if reward.shape[0] < 1:
            continue

        # argmax = np.argmax(reward[:, 0])
        print("Get the last one")
        argmax = reward.shape[0] - 1
        rewards[env] += [reward[argmax]]
        names += [algo_names[algo][0]]

    rewards[env] = np.array(rewards[env])
    colors = [[0.3, 0.3,0.3]]* len(names)
    argmax = np.argmax(rewards[env][:,0])

    colors[argmax] = 'r'
    print(colors)
    print(names, rewards[env])
    plt.bar(names, rewards[env][:,0], color=colors, yerr=rewards[env][:,1] * 1.96)
    plt.plot([-0.5, 3.5], [optimal[env], optimal[env]], "k--")

    we = expert[env]
    plt.fill_between([-0.5, 3.5], y1=[we[0]-we[1],we[0]-we[1]], y2=[we[0]+we[1],we[0]+we[1]],
        alpha=0.3, color=[0.8,0.8,0])
    plt.plot([-0.5, 3.5], [we[0],we[0]], label='Expert', color=[0.8,0.8,0])


    plt.ylim([0, optimal[env]])
    plt.yticks([0, we[0], optimal[env]])

    plt.text(2.7, optimal[env]+1.5, r'Optimal$^*$')
    plt.text(2.7, we[0]+1.5, r'Expert', color=[0.8,0.8,0])
    plt.savefig('{}_{}.png'.format(env, alg_type))
