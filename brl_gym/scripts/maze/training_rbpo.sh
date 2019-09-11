OPENAI_LOGDIR=~/models/maze-high-sensing-cost/rbpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=Maze-entropy-only-v0 --num_timesteps=5e8 --save_path=~/models/complete/maze_high-sensing-cost-rbpo --num_env=20 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=20


#echo "Finished Training Maze for Entropy-only on instance 2" | ssmtp gilwoo301@gmail.com

#sudo poweroff
