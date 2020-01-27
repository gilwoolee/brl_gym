OPENAI_LOGDIR=~/models/continuous_cartpole/rbpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=bayes-ContinuousCartPole-v0 --num_timesteps=5e8 --save_path=~/models/complete/continuous_cartpole/rbpo --num_env=10 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=5 --residual_weight=2.0
