from gym.envs.registration import register

# [GL] BPO base envs
# ---------
register(
    id='tiger-v0',
    entry_point='brl_gym.envs:Tiger',
    max_episode_steps=100,
    reward_threshold=10.0,
    nondeterministic=True
)

register(
    id='lightdark-v0',
    entry_point='brl_gym.envs:LightDark',
    max_episode_steps=30,
    reward_threshold=0.0,
    nondeterministic=True
)

register(
    id='chain-v0',
    entry_point='brl_gym.envs:Chain',
    max_episode_steps=100,
    reward_threshold=1.0,
    nondeterministic=True
)

register(
    id='semitiedchain-v0',
    entry_point='brl_gym.envs:Chain',
    kwargs={'semitied': True},
    max_episode_steps=100,
    reward_threshold=1.0,
    nondeterministic=True
)

register(
    id='rocksample-v0',
    entry_point='brl_gym.envs:RockSample',
    max_episode_steps=200,
    reward_threshold=10.0,
    nondeterministic=True
)


# [GL] BPO
# ---------
register(
    id='bayes-tiger-v0',
    entry_point='brl_gym.wrapper_envs:BayesTiger',
    max_episode_steps=20,
    reward_threshold=1.0,
    nondeterministic=True
)

register(
    id='explicit-bayes-tiger-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesTiger',
    max_episode_steps=20,
    reward_threshold=1.0,
    nondeterministic=True
)

register(
    id='bayes-lightdark-v0',
    entry_point='brl_gym.wrapper_envs:BayesLightDark',
    max_episode_steps=30,
    reward_threshold=0.0,
    nondeterministic=True
)

register(
    id='bayes-chain-v0',
    entry_point='brl_gym.wrapper_envs:BayesChain',
    max_episode_steps=100,
    reward_threshold=1.0,
    nondeterministic=True
)

register(
    id='bayes-semitiedchain-v0',
    entry_point='brl_gym.wrapper_envs:BayesChain',
    kwargs={'semitied': True},
    max_episode_steps=100,
    reward_threshold=1.0,
    nondeterministic=True
)

register(
    id='explicit-bayes-chain-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesChain',
    max_episode_steps=100,
    reward_threshold=1.0,
    nondeterministic=True
)
register(
    id='bayes-rocksample-v0',
    entry_point='brl_gym.wrapper_envs:BayesRockSample',
    max_episode_steps=500,
    reward_threshold=10.0,
    nondeterministic=True
)

register(
    id='explicit-bayes-rocksample-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesRockSample',
    max_episode_steps=100,
    reward_threshold=10.0,
    nondeterministic=True
)

register(
    id='explicit-bayes-rocksample-rock8-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesRockSampleRock8',
    max_episode_steps=100,
    reward_threshold=50.0,
    nondeterministic=True
)

register(
    id='bayes-rocksample-grid7rock8-v0',
    entry_point='brl_gym.wrapper_envs:BayesRockSampleGrid7Rock8',
    max_episode_steps=500,
    reward_threshold=100.0,
    nondeterministic=True,
)

register(
    id='bayes-herbtable-v0',
    entry_point='brl_gym.wrapper_envs:BayesHerbTable',
    max_episode_steps=500,
    reward_threshold=10.0,
    nondeterministic=True,
)

register(
    id='explicit-bayes-CartPole-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=500.0,
)

register(
    id='bayes-CartPole-v0',
    entry_point='brl_gym.wrapper_envs:BayesCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=500.0,
)

register(
    id='explicit-bayes-ContinuousCartPole-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesContinuousCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=500.0,
)

register(
    id='bayes-ContinuousCartPole-v0',
    entry_point='brl_gym.wrapper_envs:BayesContinuousCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=500.0,
)

register(
    id='ContinuousCartPole-v0',
    entry_point='brl_gym.envs:ContinuousCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=500.0,
)


register(
    id='ExplicitPusher-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitPusherEnv',
    max_episode_steps=220,
    reward_threshold=0.0,
)

register(
    id='Maze-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    max_episode_steps=500,
    reward_threshold=10,
)

register(
    id='Maze-no-entropy-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnvNoEntropyReward',
    max_episode_steps=500,
    reward_threshold=10,
)


register(
    id='Maze-expert-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnvWithExpert',
    max_episode_steps=500,
    reward_threshold=10,
)


register(
    id='Door-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesDoorsEnv',
    max_episode_steps=200,
    reward_threshold=100,
)

# [GL] MLE
# ---------
register(
    id='mle-tiger-v0',
    entry_point='brl_gym.wrapper_envs:MLETiger',
    max_episode_steps=100,
    reward_threshold=10.0,
    nondeterministic=True
)

register(
    id='mle-lightdark-v0',
    entry_point='brl_gym.wrapper_envs:MLELightDark',
    max_episode_steps=30,
    reward_threshold=0.0,
    nondeterministic=True
)

register(
    id='mle-chain-v0',
    entry_point='brl_gym.wrapper_envs:MLEChain',
    max_episode_steps=100,
    reward_threshold=1.0,
    nondeterministic=True
)

register(
    id='mle-semitiedchain-v0',
    entry_point='brl_gym.wrapper_envs:MLEChain',
    kwargs={'semitied': True},
    max_episode_steps=100,
    reward_threshold=1.0,
    nondeterministic=True
)

register(
    id='mle-rocksample-v0',
    entry_point='brl_gym.wrapper_envs:MLERockSample',
    max_episode_steps=50,
    reward_threshold=10.0,
    nondeterministic=True
)
