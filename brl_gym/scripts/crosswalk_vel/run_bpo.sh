#!/bin/bash

## BPO
python -m brl_baselines.run --alg=ppo2 --env=bayes-crosswalkvel-v0 --play --load_path=~/models/crosswalk_vel/bpo/checkpoints/ckpt-400 --num_timesteps=0 --num_trials=2500  --output=test.txt --render=True

