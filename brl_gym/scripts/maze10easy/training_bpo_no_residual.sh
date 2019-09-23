echo "Started Training Maze for BPO-expert-no-residaul on instance 3" | ssmtp gilwoo301@gmail.com

OPENAI_LOGDIR=~/models/maze/entropy_rbpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bpo_expert_no_residual --env=Maze-v0 --num_timesteps=5e8 --gamma=1 --nminibatches=20 --save_interval=50 --num_env=20 --value_network=copy --save_path=/home/gilwoo/models/maze/complete/bpo_expert_no_residual


echo "Finished Training for BPO-expert-no-residaul on instance 3" | ssmtp gilwoo301@gmail.com

sudo poweroff
