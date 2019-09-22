OPENAI_LOGDIR=~/models/doors/entropy_hidden_no_expert_input_rbpo_noent OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2 --env=Door-entropy-hidden-no-reward-v0 --num_timesteps=5e8 --save_path=~/models/complete/doors_entropy_rbpo --num_env=20 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=20


#echo "Finished Training Door-entropy-hidden-v0 for bppo2_expert on instance 3" | ssmtp gilwoo301@gmail.com

#sudo poweroff
