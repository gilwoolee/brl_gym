OPENAI_LOGDIR=~/models/test/continuous_cartpole/bpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=ppo2 --env=bayes-ContinuousCartPole-v0 --num_timesteps=5e8 --gamma=1 --nminibatches=5 --save_interval=1 --num_env=10 

