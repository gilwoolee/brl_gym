OPENAI_LOGDIR=~/models/maze10easy/upmle_ent1 OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=ppo2 --env=Maze10easy-upmle-v0 --num_timesteps=5e8 --save_path=~/models/complete/maze_upmle --num_env=8 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=8


echo "Finished Training Maze for UPMLE on instance 2" | ssmtp gilwoo301@gmail.com

sudo poweroff
