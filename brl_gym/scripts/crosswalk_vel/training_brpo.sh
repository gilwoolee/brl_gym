OPENAI_LOGDIR=~/models/crosswalk_vel/brpo_0.05 OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=bayes-crosswalkvel-v0 --num_timesteps=5e8 --save_path=~/models/complete/crosswalk_vel/brpo --num_env=20 --save_interval=1  --gamma=0.99  --residual_weight=0.05 --lr=2e-4

