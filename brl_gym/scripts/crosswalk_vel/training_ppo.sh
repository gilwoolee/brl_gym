OPENAI_LOGDIR=~/models/crosswalk/ppo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=ppo2 --env=crosswalk-v0 --num_timesteps=5e8 --gamma=0.99 --nminibatches=10 --save_interval=50 --num_env=10 --value_network=copy --save_path=/home/gilwoo/models/crosswalk/ppo
