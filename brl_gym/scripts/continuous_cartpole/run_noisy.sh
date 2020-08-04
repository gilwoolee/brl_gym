#!/bin/bash

# Run on 
source ~/virtualenv/bpo/bin/activate

# python -m brl_baselines.run --alg=bppo2_expert --env=bayes-NoisyContinuousCartPole-v0 --num_timesteps=0 --play --load_path=~/models/test/continuous_cartpole/brpo/checkpoints/ckpt-30  --num_env=1  --num_trials=100 --output=bpo_00001.txt


# python -m brl_baselines.run --alg=bppo2_expert --env=bayes-NoisyContinuousCartPoleNoiseEstimator-v0 --num_timesteps=0 --play --load_path=~/models/test/continuous_cartpole/brpo/checkpoints/ckpt-30  --num_env=1  --num_trials=100 --output=brpo_00001.txt 

# python -m brl_baselines.run --alg=bppo2_expert --env=bayes-ContinuousCartPole-v0 --num_timesteps=0 --play --load_path=~/models/test/continuous_cartpole/brpo/checkpoints/ckpt-30  --num_env=1  --num_trials=5 --output=brpo_00001.txt --render=True
