OPENAI_LOGDIR=~/scp_models/models/doors/rbpo_ent_10 OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=Door-entropy-10-v0 --num_timesteps=5e8 --save_path=~/models/complete/doors_rbpo_ent_10 --num_env=20 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=20 --load_path=~/scp_models/models/doors/rbpo_ent_10/checkpoints/04650 --update_starts_from=4651 


