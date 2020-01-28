OPENAI_LOGDIR=~/models/continuous_cartpole/mle OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=ppo2 --env=mle-ContinuousCartPole-v0 --num_timesteps=5e8 --gamma=1 --nminibatches=5 --save_interval=50 --num_env=10 --value_network=copy --save_path=/home/gilwoo/models/continuous_cartpole/mle_complete

