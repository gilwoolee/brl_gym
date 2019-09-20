#echo "Started Training Doors for RBPO on instance 3" | ssmtp gilwoo301@gmail.com
OPENAI_LOGDIR=~/models/doors/rbpo_alpha_1 OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=Door-v0 --num_timesteps=5e8 --save_path=~/models/complete/doors_rbpo_alpha_1 --num_env=20 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=20 residual_weight=1.0

#echo "Finished Training Doors for RBPO on instance 3" | ssmtp gilwoo301@gmail.com

#sudo poweroff
