#!/bin/bash

source ~/venv/brl/bin/activate

# python -m brl_baselines.run --alg=ppo2 --env=Maze-no-entropy-v0 --num_timesteps=0 --play --load_path=~/models/maze/bpo/checkpoints/00001  --num_env=1  --num_trials=500 --output=bpo_00001.txt


for i in $(seq -f "%05g" 1850 100 12200)
do
	# j=($i % 5)
	# echo $j
	# if [ $j -ne 0 ]; then
	# 	continue
	# fi

	echo $i
	python -m brl_baselines.run --alg=ppo2 --env=Maze-no-entropy-v0 --num_timesteps=0 --play --load_path=~/models/maze/bpo/checkpoints/$i  --num_env=1  --num_trials=500 --output=bpo_$i.txt
done
