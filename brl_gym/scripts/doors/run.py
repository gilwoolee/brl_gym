import os
import glob

#os.system('source ~/venv/brl/bin/activate')

rootdir = "/home/gilwoo/scp_models/models/doors/"
algos = [x[0] for x in os.walk(rootdir) if "checkpoints" in x[0]]

num_trials = 150
dry_run = False

algo_to_alg = {
    # "bpo": ["ppo2","Door-no-entropy-v0"],
    # "upmle": ["ppo2", "Door-upmle-no-entropy-v0"],
    # "expert_no_residual": ["bpo_expert_no_residual", "Door-no-entropy-v0"],
    # "rbpo": ["bppo2_expert", "Door-no-entropy-v0"],
    # "rbpo_ent_alpha_1": ["bppo2_expert", "Door-no-entropy-v0", 1.0],
    #"rbpo_noent_alpha_0.25": ["bppo2_expert", "Door-no-entropy-v0", 0.25],
    #"rbpo_noent_alpha_0.5": ["bppo2_expert", "Door-no-entropy-v0", 0.5],
    #"rbpo_noent_alpha_1": ["bppo2_expert", "Door-no-entropy-v0", 1.0],
    # "entropy_hidden_rbpo": ["bppo2_expert", "Door-entropy-hidden-no-reward-v0", 0.1],
    # "entropy_rbpo": ["bppo2_expert", "Door-entropy-only-no-reward-v0"],
    "rbpo_ent_10": ["bppo2_expert", "Door-no-entropy-v0", 0.1]
    #"entropy_hidden_no_expert_input_rbpo_noent": ["bppo2", "Door-entropy-hidden-no-reward-v0"]
}

for algo in algos:
    algname = algo.split("/")[-2]
    if algname not in algo_to_alg:
        continue
    print("--------------------")
    alg, env, alpha = algo_to_alg[algname]
    print(algo, alg)

    checkpoints = glob.glob(os.path.join(algo, "*"))
    checkpoints.sort()
    last = int(checkpoints[-1].split("/")[-1])

    outputdir = "/home/gilwoo/output/doors/"+algname
    if not os.path.exists(outputdir):
        print("Make ", outputdir)
        os.makedirs(outputdir)


    for i in [4700]:#range(4700, last, 100):#m]:#, 3000, 3050]:#[1] + list(range(100, last, 100)):
        outputfile = "{}/{}.txt".format(outputdir, str(i).zfill(5))
        # if os.path.exists(outputfile):
        #    continue

        cmd = "python -m brl_baselines.run --alg={} --env={} --num_timesteps=0 --play --load_path={}/{}  --num_env=1  --num_trials={} --output={}/{}.txt --residual_weight={}".format(alg, env, algo, str(i).zfill(5), num_trials, outputdir, str(i).zfill(5), alpha)
        print(cmd)
        if not dry_run:
            os.system(cmd)

#        import sys; sys.exit(0)
