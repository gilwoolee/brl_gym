#!/bin/bash

## BRPO
python -m brl_baselines.run --alg=bppo2_expert --env=bayes-crosswalkvel-v0 --play --load_path=~/models/crosswalk_vel/brpo/checkpoints --num_timesteps=0 --num_trials=5  --residual_weight=0.5 --output=test.txt --render=True 