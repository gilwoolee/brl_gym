echo "Started Training Doors for bpo_expert_no_residual on instance 5" | ssmtp gilwoo301@gmail.com
OPENAI_LOGDIR=~/models/doors/bpo_expert_no_residual OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bpo_expert_no_residual --env=Door-v0 --num_timesteps=5e8 --gamma=1 --nminibatches=20 --save_interval=50 --num_env=20 --value_network=copy  save_path=/home/gilwoo/models/doors/complete/bpo_expert_no_residual

echo "Finished Training Doors for bpo_expert_no_residual on instance 5" | ssmtp gilwoo301@gmail.com

sudo poweroff
