OPENAI_LOGDIR=~/models/lightdark/bpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=ppo2 --env=bayes-lightdark-v0 --num_timesteps=5e8 --gamma=1 --nminibatches=5 --save_interval=50 --num_env=5 --value_network=copy --save_path=/home/gilwoo/models/lightdark/bpo

