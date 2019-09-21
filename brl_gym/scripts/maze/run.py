import os
import glob

#os.system('source ~/venv/brl/bin/activate')

rootdir = "/home/gilwoo/models/maze/"
algos = [x[0] for x in os.walk(rootdir) if "checkpoints" in x[0]]

num_trials = 500
dry_run = False

algo_to_alg = {
    # "single_expert_rbpo": ["bppo2_expert", "Maze-entropy-hidden-no-reward-v0"],
    # "entropy_hidden_rbpo": ["bppo2_expert", "Maze-entropy-hidden-no-reward-v0"],
    #"rbpo_stronger_expert": ["bppo2_expert", "Maze-no-entropy-v0"],
    # "entropy_rbpo": ["bppo2_expert", "Maze-entropy-only-no-reward-v0"],
    "bpo_noent": ["ppo2","Maze-no-entropy-v0", 0.0],
    # "upmle": ["ppo2", "Maze-upmle-no-reward-v0"],
    # "expert_no_residual": ["bpo_expert_no_residual", "Maze-no-entropy-v0"],
    # "noentropy_rbpo": ["bppo2_expert", "Maze-no-entropy-v0"],
    # "rbpo_hidden_belief_no_ent_reward": ["bppo2_expert", "Maze-entropy-hidden-no-reward-v0"],
    # "rbpo-noent-alpha-1.0":["bppo2_expert", "Maze-no-entropy-v0", 1.0]
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

    outputdir = "/home/gilwoo/output/maze/"+algname
    if not os.path.exists(outputdir):
        print("Make ", outputdir)
        os.makedirs(outputdir)

    for i in [1] + list(range(100, last, 100)):
        outputfile = "{}/{}.txt".format(outputdir, str(i).zfill(5))
        if os.path.exists(outputfile):
            continue

        if alg.startswith("ppo2"):
           cmd = "python -m brl_baselines.run --alg={} --env={} --num_timesteps=0 --play --load_path={}/{}  --num_env=1  --num_trials={} --output={}/{}.txt".format(alg, env, algo, str(i).zfill(5), num_trials, outputdir, str(i).zfill(5))

        else:

           cmd = "python -m brl_baselines.run --alg={} --env={} --num_timesteps=0 --play --load_path={}/{}  --num_env=1  --num_trials={} --output={}/{}.txt --residual_weight={}".format(alg, env, algo, str(i).zfill(5), num_trials, outputdir, str(i).zfill(5),alpha)

        print(cmd)
        if not dry_run:
            os.system(cmd)

#        import sys; sys.exit(0)
