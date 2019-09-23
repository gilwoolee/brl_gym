OPENAI_LOGDIR=~/models/maze10easy/rbpo_noent_alpha_1 OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=Maze10easy-noent-v0 --num_timesteps=5e8 --save_path=~/models/complete/maze10_rbpo --num_env=8 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=8 --residual_weight=1.0



#echo "Finished Training Maze for Entropy-only on instance 2" | ssmtp gilwoo301@gmail.com

#sudo poweroff