 OPENAI_LOGDIR=~/models/door/rbpo_learnablebf_alpha_0.1 OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert_learned_bf --env=Door-LearnableBF-noent-v0 --num_timesteps=5e8 --save_path=~/models/complete/door_learnablebf_rbpo --num_env=4 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=4 --residual_weight=0.1