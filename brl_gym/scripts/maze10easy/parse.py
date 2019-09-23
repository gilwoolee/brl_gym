import glob
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib; matplotlib.use('PDF')

matplotlib.rc('font', family='serif', size=25)
matplotlib.rc('text', usetex=True)

from matplotlib import pyplot as plt

algo_to_alg = {
    "rbpo_ent_100_alpha_1": ["E100-A1",'c'],
    # "rbpo_ent_100_alpha_0.1": ["E100-A0.1",'b'],
    # "rbpo_ent_100_alpha_0.5": ["E100-A0.5",[0.8,0.7,0.7]],

    # "rbpo_ent10_alpha_1": ["E10-A1",'r'],
    # "rbpo_ent10_alpha_0.1": ["E10-A0.5",'y'],
    "rbpo_ent10_alpha_0.25": ["E10-A0.25",'m'],
    "rbpo_ent10_alpha_0.5": ["E10-A0.5",'k'],

    "rbpo_ent1_alpha_1": ["E1-A1",[1.0,0.0,0.8]],

    "rbpo_noent_alpha_1": ["E0-A1",'g'],
}


# ent weights
# algo_to_alg = {
#     "rbpo_ent_100_alpha_1": ["100",'#FA1900'],
#     "rbpo_ent10_alpha_1": ["10",'#BFBFBF'],
#     "rbpo_noent_alpha_1": ["0",'#8C7F70'],
# }
# name = "ent"
# algnames = ['rbpo_noent_alpha_1', 'rbpo_ent10_alpha_1',
#             'rbpo_ent_100_alpha_1']

# ent input
algo_to_alg = {
    "rbpo_ent_100_alpha_1": ["Belief",'#FA1900'],
    # "rbpo_ent_100_alpha_0.1": ["E100-A0.1",'b'],
    # "rbpo_ent_100_alpha_0.5": ["E100-A0.5",[0.8,0.7,0.7]],

    # "rbpo_ent10_alpha_1": ["E10-A1",'r'],
    # "rbpo_ent10_alpha_0.1": ["E10-A0.5",'y'],
    "rbpo_ent_hidden_ent100_alpha_1": ["None",'#504E75'],
    "rbpo_ent_input_ent100_alpha_1": ["Entropy",'#698F6E']
}
name = "ent_input"
algnames = ['rbpo_ent_hidden_ent100_alpha_1',
            'rbpo_ent_input_ent100_alpha_1',
            'rbpo_ent_100_alpha_1']

# baselines
# algo_to_alg = {
#     "bpo_ent100": ["BPO",'#8C7F70'],
#     "upmle_ent_100": ["UPMLE",'#F2D39B'],
#     "rbpo_ent_100_alpha_1": [r'\bf{RBPO}','#FA1900']
# }
# name = "baseline"
# algnames = ['bpo_ent100', 'upmle_ent_100',
#             'rbpo_ent_100_alpha_1']


stat = dict()

fig, ax = plt.subplots(figsize=(8,6))
env = "maze10"

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

max_step = 7000

we = [80.73, 23.68*1.96]
random = -500 * 0.9 + 500 * 0.1
maximum = 500

plt.plot([0, max_step], [500, 500], 'k--', lw=4)
# plt.text(max_step + 40, maximum - 10, r'Optimal$^*$', color='k')

plt.fill_between([0,max_step], y1=[we[0]-we[1],we[0]-we[1]],
    y2=[we[0]+we[1],we[0]+we[1]], alpha=0.3, color="#597223")
plt.plot([0, max_step], [we[0],we[0]],  color='#597223', lw=4)
# plt.text(max_step + 40, we[0] - 10, r'Expert', color='#597223')

if name == "baseline":
    # plt.ylim(random, 500)
    plt.yticks([random, 0, we[0], maximum])

    plt.plot([0, max_step], [random, random], '--', color="#878787", lw=4)
    # plt.text(max_step + 40, random - 10, r'Random', color='#878787')

else:
    # plt.ylim(we[0]-we[1], 500)
    plt.ylim(0, 500)

plt.xlim(0, max_step)
plt.xticks([6000])
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
        rewards += [(timestep, np.mean(data[:, 1]), 1.96*np.std(data[:,1])/np.sqrt(data.shape[0]))]

    rewards = np.array(rewards)
    print(rewards[:5])
    stat[pr] = rewards
    plt.fill_between(rewards[:,0], y1=rewards[:,1] - rewards[:,2], y2=rewards[:max_step,1]+rewards[:max_step,2],
        alpha=0.3, color=algo_to_alg[pr][1])
    plt.plot(rewards[:,0], rewards[:,1], label=algo_to_alg[pr][0], lw=4, color=algo_to_alg[pr][1])

    if timestep < max_step:
        # extend the line
        plt.plot([rewards[-1,0], max_step], [rewards[-1,1], rewards[-1,1]],
            '--', lw=4, color=algo_to_alg[pr][1])


legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
          ncol=3, frameon=False)


plt.savefig('{}_{}.pdf'.format(env, name), bbox_inches='tight')
print('{}_{}.pdf'.format(env, name))
# plt.show()



