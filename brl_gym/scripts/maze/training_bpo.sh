OPENAI_LOGDIR=~/models/maze/bpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=ppo2 --env=Maze-v0 --num_timesteps=5e8 --save_path=~/models/complete/maze_bpo --num_env=20 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=20