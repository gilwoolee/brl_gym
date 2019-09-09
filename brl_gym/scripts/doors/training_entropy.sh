echo "Started Training Doors for Entropy-onpy on instance 3" | ssmtp gilwoo301@gmail.com
OPENAI_LOGDIR=~/models/doors/entropy_rbpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=Door-entropy-only-v0 --num_timesteps=5e8 --save_path=~/models/complete/doors_entropy_rbpo --num_env=20 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=20


echo "Finished Training Doors for Entropy-only on instance 3" | ssmtp gilwoo301@gmail.com

#sudo poweroff
