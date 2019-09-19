#!/bin/bash

source ~/venv/brl/bin/activate

python -m brl_baselines.run --alg=bppo2_expert --env=Maze-no-entropy-v0 --num_timesteps=0 --play --load_path=~/models/maze/bppo_expert_corrective/checkpoints/00001  --num_env=1  --num_trials=100 --output=expert_00001.txt


for i in $(seq -f "%05g" 100 100 1600)
do
	echo $i
	python -m brl_baselines.run --alg=bppo2_expert --env=Maze-no-entropy-v0 --num_timesteps=0 --play --load_path=~/models/maze/bppo_expert_corrective/checkpoints/$i  --num_env=1  --num_trials=100 --output=expert_$i.txt
done
