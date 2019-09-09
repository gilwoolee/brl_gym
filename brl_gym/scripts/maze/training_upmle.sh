echo "Started Training Maze for UPMLE on instance 2" | ssmtp gilwoo301@gmail.com
OPENAI_LOGDIR=~/models/maze/upmle OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=ppo2 --env=Maze-upmle-v0 --num_timesteps=5e8 --save_path=~/models/complete/maze_upmle --num_env=20 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=20


echo "Finished Training Maze for UPMLE on instance 2" | ssmtp gilwoo301@gmail.com

sudo poweroff
