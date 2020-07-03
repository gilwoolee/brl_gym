OPENAI_LOGDIR=~/models/crosswalk_vel/brpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=bayes-crosswalkvel-v0 --num_timesteps=5e8 --save_path=~/models/complete/crosswalk_vel/brpo --num_env=10 --save_interval=50  --gamma=0.99  --nminibatches=10 --residual_weight=0.5