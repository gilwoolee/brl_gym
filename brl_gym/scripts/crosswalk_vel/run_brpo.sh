#!/bin/bash

## BRPO
python -m brl_baselines.run --alg=bppo2_expert --env=bayes-crosswalkvel-v0 --play --load_path=~/models/crosswalk_vel/brpo_0.2/checkpoints/chpt-80 --num_timesteps=0 --num_trials=5  --residual_weight=0.2 --output=test.txt --render=True
