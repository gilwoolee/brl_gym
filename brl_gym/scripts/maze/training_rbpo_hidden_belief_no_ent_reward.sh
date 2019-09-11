echo "Started Training Maze-hidden-belief-no-ent-reward for RBPO on instance 9" | ssmtp gilwoo301@gmail.com
OPENAI_LOGDIR=~/models/maze/rbpo_hidden_belief_no_ent_reward OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=Maze-hidden-belief-no-ent-reward-v0 --num_timesteps=5e8 --save_path=~/models/complete/maze_rbpo_hidden_belief_no_ent_reward --num_env=20 --save_interval=50  --gamma=1 --value_network=copy  --nminibatches=20


echo "Finished Training Maze-hidden-belief-no-ent-reward for RBPO on instance 9" | ssmtp gilwoo301@gmail.com

#i1o poweroff
