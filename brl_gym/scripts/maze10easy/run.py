import os
import glob

#os.system('source ~/venv/brl/bin/activate')

rootdir = "/home/gilwoo/models/maze10easy/"
algos = [x[0] for x in os.walk(rootdir) if "checkpoints" in x[0]]

num_trials = 500
dry_run = False

algo_to_alg = {
#        "rbpo_noent_alpha_1":["bppo2_expert", "Maze10easy-noent-v0", 1.0]
        "entropy_hidden_no_expert_input_rbpo_noent":["bppo2", "Maze10easy-entropy-hidden-noent-v0", 1.0]
    # "single_expert_rbpo": ["bppo2_expert", "Maze-entropy-hidden-no-reward-v0"],
    # "entropy_hidden_rbpo": ["bppo2_expert", "Maze-entropy-hidden-no-reward-v0"],
    #"rbpo_stronger_expert": ["bppo2_expert", "Maze-no-entropy-v0"],
    # "entropy_rbpo": ["bppo2_expert", "Maze-entropy-only-no-reward-v0"],
    # "bpo": ["ppo2","Maze-no-entropy-v0"],
    # "upmle": ["ppo2", "Maze-upmle-no-reward-v0"],
    # "expert_no_residual": ["bpo_expert_no_residual", "Maze-no-entropy-v0"],
    # "noentropy_rbpo": ["bppo2_expert", "Maze-no-entropy-v0"],
    # "rbpo_hidden_belief_no_ent_reward": ["bppo2_expert", "Maze-entropy-hidden-no-reward-v0"],
}

for algo in algos:
    algname = algo.split("/")[-2]
    if algname not in algo_to_alg:
        continue
    print("--------------------")
    alg, env, alpha = algo_to_alg[algname]
    print(algo, alg, alpha)

    checkpoints = glob.glob(os.path.join(algo, "*"))
    checkpoints.sort()
    last = int(checkpoints[-1].split("/")[-1])

    outputdir = "/home/gilwoo/output/maze10easy/"+algname
    if not os.path.exists(outputdir):
        print("Make ", outputdir)
        os.makedirs(outputdir)

    for i in [1] + list(range(100, last, 100)):
        outputfile = "{}/{}.txt".format(outputdir, str(i).zfill(5))
        if os.path.exists(outputfile):
            continue

        cmd = "python -m brl_baselines.run --alg={} --env={} --num_timesteps=0 --play --load_path={}/{}  --num_env=1  --num_trials={} --output={}/{}.txt --residual_weight={}".format(alg, env, algo, str(i).zfill(5), num_trials, outputdir, str(i).zfill(5), alpha)
        print(cmd)
        if not dry_run:
            os.system(cmd)

#        import sys; sys.exit(0)
