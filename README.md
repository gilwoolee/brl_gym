# BRL Gym

BRL Gym is an extension of OpenAI Gym for Bayesian RL problems. 

BRL Gym contains a set of Bayesian Environments. Each BayesEnv is composed of two components: Env and Estimator. 

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
