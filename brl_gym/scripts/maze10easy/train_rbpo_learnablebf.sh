OPENAI_LOGDIR=~/models/maze10/rbpo_learnablebf_alpha_0.1 OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert_learned_bf --env=Maze10easy-LearnableBF-noent-v0 --num_timesteps=5e8 --save_path=~/models/complete/maze10easy_learnablebf_rbpo --num_env=1 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=10 --residual_weight=0.1
