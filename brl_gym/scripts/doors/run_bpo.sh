#!/bin/bash

# source ~/venv/brl/bin/activate

python -m brl_baselines.run --alg=ppo2 --env=Door-no-entropy-v0 --num_timesteps=0 --play --load_path=~/models/doors/bpo/checkpoints/00001  --num_env=1  --num_trials=100 --output=bpo_00001.txt

for i in $(seq -f "%05g" 50 50 3650)
do
	echo $i
	python -m brl_baselines.run --alg=ppo2 --env=Door-no-entropy-v0 --num_timesteps=0 --play --load_path=~/models/doors/bpo/checkpoints/$i  --num_env=1  --num_trials=100 --output=bpo_$i.txt
done
