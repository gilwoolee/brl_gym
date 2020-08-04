# OPENAI_LOGDIR=~/models/test/doors/brpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=Door-v0 --num_timesteps=5e8  --num_env=20 --save_interval=1  --gamma=1   --nminibatches=20

OPENAI_LOGDIR=~/models/test/doors_noisy/brpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=DoorNoisy-v0 --num_timesteps=5e8  --num_env=20 --save_interval=1  --gamma=1   --nminibatches=20
