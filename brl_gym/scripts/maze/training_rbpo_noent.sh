OPENAI_LOGDIR=~/models/maze/rbpo-noent-alpha-1.0 OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=Maze-no-entropy-v0 --num_timesteps=5e8 --save_path=~/models/complete/maze_noent-rbpo-alpha1.0 --num_env=20 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=20 --residual_weight=1.0


echo "Finished Training Maze for noent-reward/alpha1 on instance 6" | ssmtp gilwoo301@gmail.com

#sudo poweroff
