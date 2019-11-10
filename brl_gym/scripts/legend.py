# fmt: off
import matplotlib; matplotlib.use('PDF')
matplotlib.rc('font', family='serif', serif=['CMU Serif'])
matplotlib.rc('text', usetex=True)
# fmt: on
import matplotlib.pyplot as plt
import os

import brl_gym.scripts.colors as colors


labels = {
    'random': dict(label=r'\textsc{Random}', color=colors.random_color),
    'optimal': dict(label=r'\textsc{Optimal}', color=colors.max_color),
    'expert': dict(label=r'\textsc{Ensemble}', color=colors.expert_color),
    'brpo': dict(label=r'\bf{BRPO}', color=colors.brpo_color[colors.EMPHASIS]),
    'upmle': dict(label='UPMLE', color=colors.upmle_color[colors.EMPHASIS]),
    'bpo': dict(label='BPO', color=colors.bpo_color[colors.EMPHASIS]),

    'b+e': dict(label=r'\bf{Belief + Recommendation}', color=colors.brpo_color[colors.EMPHASIS]),
    'e': dict(label='Recommendation only', color=colors.ensemble_color[colors.EMPHASIS]),
    'none': dict(label='None', color=colors.none_color[colors.EMPHASIS]),

    "0": dict(label=r'\bf{$\epsilon=0$}', color=colors.brpo_color[colors.EMPHASIS]),
    "10": dict(label=r'$\epsilon=10$', color=colors.ent_reward_10[colors.EMPHASIS]),
    "100": dict(label=r'$\epsilon=100$', color=colors.ent_reward_100[colors.EMPHASIS]),


}

label_order = {
"baseline": ["brpo", "upmle", "bpo", "expert", "random", "optimal"],
"ent_input": ["b+e","e", "expert", "optimal"],
"ent_reward": ["0", "10", "100", "expert", "optimal"]
}


if __name__ == '__main__':
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    for key in label_order:
        labels_in_key = dict()
        for x in label_order[key]:
            labels_in_key[x] = labels[x]

        # fig = plt.figure(figsize=(4.5, 0.5))
        fig = plt.figure(figsize=(8.0, 0.5))
        patches = [
            mpatches.Patch(**labels_in_key[label])
            for label in label_order[key]]

        plt.axis('off')
        plt.legend(patches,
            [labels_in_key[label]['label'] for label in label_order[key]],
            loc='center', frameon=False, ncol=len(labels_in_key.values()))
        plt.savefig('legend_{}.pdf'.format(key))