# Using the correct bayes filter, learn from scratch without experts. 
OPENAI_LOGDIR=~/models/test/continuous_cartpole_noisy/bpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=ppo2 --env=bayes-NoisyContinuousCartPoleNoiseEstimator-v0 --num_timesteps=5e8 --num_env=4 --save_interval=1  --gamma=1.0

OPENAI_LOGDIR=~/models/test/continuous_cartpole/brpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=bayes-ContinuousCartPole-v0 --num_timesteps=5e8 --num_env=4 --save_interval=1  --gamma=1.0  --residual_weight=0.1 --lr=1e-4

# Not so ideal scenario using the original (incorrect) bayes filter, learn from scratch. 
# The expert mixture is incorrect because the belief is incorrect, but the residual policy can make up for it.
OPENAI_LOGDIR=~/models/test/continuous_cartpole_noisy_incorrect_bf/brpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=bayes-NoisyContinuousCartPole-v0 --num_timesteps=5e8 --num_env=4 --save_interval=1  --gamma=1.0  --residual_weight=0.1 --lr=1e-4

# Not so ideal scenario using the original (incorrect) bayes filter on the (ideal) "correct" env, learn from brpo policy learned on the incorrect noise-free env
OPENAI_LOGDIR=~/models/test/continuous_cartpole_noisy_incorrect_bf_from_learned_policy/brpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=bayes-NoisyContinuousCartPole-v0 --num_timesteps=5e8 --num_env=4 --save_interval=1  --gamma=1.0  --residual_weight=0.1 --load_path=~/models/test/continuous_cartpole/brpo/checkpoints/ckpt-32 --lr=1e-4

# Most ideal, using the correct bayes filter on correct env, learn from trained brpo using experts. 
OPENAI_LOGDIR=~/models/test/continuous_cartpole_noisy_from_trained/brpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=bayes-NoisyContinuousCartPoleNoiseEstimator-v0 --num_timesteps=5e8 --num_env=4 --save_interval=1  --gamma=1.0  --residual_weight=0.1 --load_path=~/models/test/continuous_cartpole/brpo/checkpoints/ckpt-32 --lr=1e-4

# "Ideal" scenario using the correct bayes filter, learn from scratch using experts. 
OPENAI_LOGDIR=~/models/test/continuous_cartpole_noisy_from_scratch/brpo OPENAI_LOG_FORMAT=tensorboard python -m brl_baselines.run --alg=bppo2_expert --env=bayes-NoisyContinuousCartPoleNoiseEstimator-v0 --num_timesteps=5e8 --num_env=8 --save_interval=1  --gamma=1.0  --residual_weight=0.5 --lr=1e-4

