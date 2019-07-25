# Environments

We provide a few environments with latent variables or noisy state observations. The environments are paired with estimators in `brl_gym.estimators` and wrapped into Bayesian environments in `brl_gym.wrapper_envs`.


### Rocksample

RockSample is a POMDP environment originally proposed by [HSVI](https://arxiv.org/pdf/1207.4166.pdf).

It has three types of actions, MOVE, SENSE, SAMPLE. The goal of the agent is to sample only good rocks along the way to the goal region (`y=6`). Upon sensing a particular rock, the agent receives a noisy observation of whether the rock is good (G) or bad (B).

To run Rocksample with a random agent, run `python bayesian_rl/scripts/rocksample/run_rocksample_random.py`, which generates snapshots of images per action.



### Chain


### LightDark


### Tiger
