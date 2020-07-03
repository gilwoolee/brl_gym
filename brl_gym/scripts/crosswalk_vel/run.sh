#!/bin/bash

## BRPO
# python -m brl_baselines.run --alg=bppo2_expert --env=bayes-crosswalk-v0 --play --load_path=~/models/crosswalk/rbpo/checkpoints/00001 --num_timesteps=0 --num_trials=100  --residual_weight=0.5 --output=data/brpo/brpo_00001.txt
# for i in $(seq -f "%05g" 1050 50 8000)
# do
# 	echo $i
# 	python -m brl_baselines.run --alg=bppo2_expert --env=bayes-crosswalk-v0 --play --load_path=~/models/crosswalk/rbpo/checkpoints/$i --num_timesteps=0 --num_trials=100  --residual_weight=0.5 --output=data/brpo/brpo_$i.txt
# done


## BPO
# python -m brl_baselines.run --alg=ppo2 --env=bayes-crosswalk-v0 --play --load_path=~/models/crosswalk/bpo/checkpoints/00001 --num_timesteps=0 --num_trials=100  --output=data/bpo/bpo_00001.txt
# for i in $(seq -f "%05g" 50 50 8000)
# do
# 	echo $i
# 	python -m brl_baselines.run --alg=ppo2 --env=bayes-crosswalk-v0 --play --load_path=~/models/crosswalk/bpo/checkpoints/$i --num_timesteps=0 --num_trials=100  --output=data/bpo/bpo_$i.txt
# done

## UPMLE
python -m brl_baselines.run --alg=ppo2 --env=bayes-crosswalk-v0 --play --load_path=~/models/crosswalk/mle/checkpoints/00001 --num_timesteps=0 --num_trials=100  --output=data/mle/mle_00001.txt
for i in $(seq -f "%05g" 50 50 8000)
do
	echo $i
	python -m brl_baselines.run --alg=ppo2 --env=bayes-crosswalk-v0 --play --load_path=~/models/crosswalk/mle/checkpoints/$i --num_timesteps=0 --num_trials=100  --output=data/mle/mle_$i.txt
done