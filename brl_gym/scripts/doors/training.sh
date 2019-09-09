echo "Started Training for UPMLE" | ssmtp gilwoo301@gmail.com

OPENAI_LOGDIR=~/models/doors/upmle OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=ppo2 --env=Door-upmle-v0 --num_timesteps=5e8 --save_path=~/models/door/upmle_model --num_env=20 --save_interval=50 --nminibatches=20 --gamma=1 --value_network=copy

echo "Finished Training for UPMLE" | ssmtp gilwoo301@gmail.com

sudo poweroff
