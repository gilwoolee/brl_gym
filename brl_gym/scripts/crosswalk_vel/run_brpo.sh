#!/bin/bash

## BRPO
python -m brl_baselines.run --alg=bppo2_expert --env=bayes-crosswalkvel-v0 --play --load_path=~/models/crosswalk_vel/brpo_0.1/checkpoints/chpt-62 --num_timesteps=0 --num_trials=500  --residual_weight=0.05 --output=test.txt --render=False

