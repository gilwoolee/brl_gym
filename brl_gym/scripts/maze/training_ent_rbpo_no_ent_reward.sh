OPENAI_LOGDIR=~/models/maze/entropy_rbpo_no_ent_reward OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=Maze-entropy-only-no-reward-v0 --num_timesteps=5e8 --save_path=~/models/complete/maze_entropy_rbpo --num_env=20 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=20



#sudo poweroff
