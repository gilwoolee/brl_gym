#!/bin/bash

## BRPO
python -m brl_baselines.run --alg=bppo2_expert --env=bayes-crosswalkvel-v0 --play --load_path=~/models/crosswalk_vel/brpo_0.1/checkpoints/ckpt-800 --num_timesteps=0 --num_trials=50  --residual_weight=0.1 --output=test.txt --render=True

