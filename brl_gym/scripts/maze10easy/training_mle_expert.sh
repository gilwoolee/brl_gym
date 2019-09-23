echo "Started Training Maze for Entropy-onpy on instance 2" | ssmtp gilwoo301@gmail.com

OPENAI_LOGDIR=~/models/maze/single_expert_rbpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=Maze-entropy-hidden-v0 --num_timesteps=5e8 --save_path=~/models/complete/maze_rbpo_mle --num_env=20 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=20 --mle=True


echo "Finished Training Maze for Entropy-only on instance 2" | ssmtp gilwoo301@gmail.com

# sudo poweroff
