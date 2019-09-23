import os
import glob

#os.system('source ~/venv/brl/bin/activate')

rootdir = "/home/gilwoo/models/doors/"
algos = [x[0] for x in os.walk(rootdir) if "checkpoints" in x[0]]

num_trials = 250
dry_run = False

algo_to_alg = {
    # "bpo": ["ppo2","Door-no-entropy-v0"],
    # "upmle": ["ppo2", "Door-upmle-no-entropy-v0"],
    # "expert_no_residual": ["bpo_expert_no_residual", "Door-no-entropy-v0"],
    # "rbpo": ["bppo2_expert", "Door-no-entropy-v0"],
    # "entropy_hidden_rbpo": ["bppo2_expert", "Door-entropy-hidden-no-reward-v0"],
    # "entropy_rbpo": ["bppo2_expert", "Door-entropy-only-no-reward-v0"],
    #"rbpo_ent_100": ["bppo2_expert", "Door-no-entropy-v0"]
    "entropy_hidden_no_expert_input_rbpo_noent": ["bppo2", "Door-entropy-hidden-no-reward-v0"]
}

for algo in algos:
    algname = algo.split("/")[-2]
    if algname not in algo_to_alg:
        continue
    print("--------------------")
    alg, env = algo_to_alg[algname]
    print(algo, alg)

    checkpoints = glob.glob(os.path.join(algo, "*"))
    checkpoints.sort()
    last = int(checkpoints[-1].split("/")[-1])

    outputdir = "/home/gilwoo/output/doors/"+algname
    if not os.path.exists(outputdir):
        print("Make ", outputdir)
        os.makedirs(outputdir)

    for i in [1]+ list(range(100, last, 100)):
        outputfile = "{}/{}.txt".format(outputdir, str(i).zfill(5))
        if os.path.exists(outputfile):
            continue

        cmd = "python -m brl_baselines.run --alg={} --env={} --num_timesteps=0 --play --load_path={}/{}  --num_env=1  --num_trials={} --output={}/{}.txt".format(alg, env, algo, str(i).zfill(5), num_trials, outputdir, str(i).zfill(5))
        print(cmd)
        if not dry_run:
            os.system(cmd)

#        import sys; sys.exit(0)
