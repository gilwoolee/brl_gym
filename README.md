This repository is under heavy development. Not all environment may work as expected. Please email [Gilwoo Lee](mailto:gilwoo301@gmail.com) for any questions regarding the environments.

# BRL Gym

BRL Gym is an extension of OpenAI Gym for Bayesian RL problems. It contains a set of Bayesian Environments. Each BayesEnv is composed of two components: Env and Estimator. 

Env is the typical OpenAI Gym env with observation space and action space. Env typically has unobservable latent parameters which define the dynamics of the environment, e.g. mass and length of the pole for CartPole. 

Estimator is a Bayes Filter over the latent models. It takes in latest (action, observation) and updates the belief over the latent models. The belief should be represented as a fixed vector (e.g. Catgegorical distribution, Gaussian Distribution). While we recommend using an analytically built Bayes Filter, an LSTM can be used to replace the Bayes Filter.

# Installation
BRL Gym builds upon OpenAI Gym 0.17.2. It mostly just borrows the gym structure. It runs on Python 3.

```bash
python -m pip install -e .
```

# Using BRL Gym
You can load a BayesEnv much like how you'd load a Gym env:
```
import gym, brl_gym
env = gym.make('bayes-CartPole-v0')
obs = env.reset()
```

# Overview

BRL environment (`BayesEnv`) is a wrapper around a regular OpenAI Gym environment. It internally maintains a Gym `environment` and a Bayes filter (`estimator`). The Gym `env` is assumed to have a latent state, which the `estimator` approximates at every step from the observations. For example, `Tiger` is an environment with a tiger hidden either behind the left or right door. Upon every `Listen` action, it returns a noisy observation of `Sound.SOUND_LEFT` or `Sound.SOUND_RIGHT`, with error probability of 0.15. (With probability 0.15, the observation is incorrect.)

```python
>>> from brl_gym.envs.tiger import Tiger
>>> env = Tiger()
>>> o = env.reset() # Returns Sound.SOUND_NONE
>>> o, _, _, info = env.step(Action.LISTEN) # returns `Sound.SOUND_LEFT` with p=0.85 if Tiger is behind the left door.
```

The `estimator` initially sets the belief of tiger location to be `[0.5, 0.5]`. At every observation, it updates the posterior according to the Bayes rule.

```python
>>> from brl_gym.estimators.bayes_tiger_estimator import BayesTigerEstimator
>>> estimator = BayesTigerEstimator()
>>> belief = estimator.reset() # [0.5, 0.5]
>>> belief = estimator.estimate(Action.LISTEN, o, info) # [0.85, 0.15] if o = Sound.SOUND_LEFT
```

BRL env is a wrapper of `env` and `estimator`. At every step, it concatenates the observation with the belief.

```python
>>> from brl_gym.wrapper_envs.wrapper_tiger import BayesTigerEnv
>>> benv = BayesTigerEnv()
>>> o = benv.reset() # Returns [Sound.SOUND_NONE, 0.5, 0.5]
>>> o, _, _, info = env.step(Action.LISTEN)
```

If the first observation was `Sound.SOUND_LEFT`, upon taking `Action.LISTEN`, it returns `[Sound.SOUND_LEFT, 0.85, 0.15]`. For discrete observations, we found one-hot encodings to be more useful for training, so it represents `Sound.SOUND_LEFT` as one-hot vector in the `BayesTigerEnv`.

It is not mandatory that the last observation is part of the output of the Bayes env. It is up to the implementer to design the output of `BayesEnv`. In this particular case, it is in fact sufficient to just have the latest belief as the output of the BRL env, since `belief` encodes the last observation. In the more general BRL setting, the assumption is that the observation out of the wrapped environment is the (observable) `state` of the system and `belief` represents the distribution over the latent part of the system.

BPO assumes that the output of the Bayes env contains both state and belief, which is why `obs_dim` is one of the required parameters for runnign BPO. If only `belief` is passed in, you can just run `ppo2` as is.

## Constructing a new BRL env

Assuming that you have a Gym environment, what you need to implement is a Bayes filter and a Bayes environment which wrapps the env and the filter.

1. Construct a Bayes filter
   A Bayes filter should support `reset`, `estimate`, `get_belief` methods of `brl_gym.estimators.Estimator` class. Its `estimate` method takes the last `action`, `observation` and `info`. Technically the Bayes filter should just use `observation`, but it can use `info` to get access to other observable information.

2. Construct a Bayes env
   You can either inherit `brl_gym.wrapper_envs.bayes_env.BayesEnv` or implement your own wrapper. Bayes env should also be a Gym environment, supporting `reset`, `step` and other methods such as `render`. It internally contains the (raw) environment that it interacts with. Upon reset, it should *reset* the internal environment and the Bayes filter. In the BRL literature it is assumed that the reset resets the latent state as well. See `brl_gym.wrapper_envs.bayes_tiger_env` for an example.

