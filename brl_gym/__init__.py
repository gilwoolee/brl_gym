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

register(
    id='wam-v0',
    entry_point='brl_gym.envs:WamEnv',
    max_episode_steps=200,
    reward_threshold=1.0,
    nondeterministic=True)



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
    id='bayes-WamFindObj-v0',
    entry_point='brl_gym.wrapper_envs:BayesWamFindObj',
    max_episode_steps=500,
    reward_threshold=10.0
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
    id='Maze-LearnableBF-noent-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    max_episode_steps=500,
    reward_threshold=500,
    kwargs={
    "entropy_weight":0.0,
    "reward_entropy":False,
    "maze_type":4,
    "learnable_bf":True
    },
)

register(
    id='Maze-slow-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    kwargs={"maze_slow":True},
    max_episode_steps=10000,
    reward_threshold=10,
)
register(
    id='Maze10-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    kwargs={"maze_type":10, "entropy_weight":1.0},
    max_episode_steps=750,
    reward_threshold=500,
)

register(
    id='MazeCont-v0',
    entry_point='brl_gym.wrapper_envs:BayesMazeContinuousEnv',
    kwargs={"entropy_weight":1.0},
    max_episode_steps=750,
    reward_threshold=500
)

register(
    id='Maze10easy-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    kwargs={"maze_type":10, "entropy_weight":1.0, 'difficulty':'easy'},
    max_episode_steps=1000,
    reward_threshold=500,
)


register(
    id='Maze10easy-slow-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    kwargs={"maze_slow":True, "maze_type":10, "difficulty":"easy"},
    max_episode_steps=10000,
    reward_threshold=500,
)

register(
    id='Maze10easy-LearnableBF-noent-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    max_episode_steps=1000,
    reward_threshold=500,
    kwargs={
    "entropy_weight":0.0,
    "reward_entropy":False,
    "maze_type":10,
    "difficulty":"easy",
    "learnable_bf":True
    },
)

register(
    id='Maze10-ent-10-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    kwargs={"maze_type":10, "entropy_weight":10.0},
    max_episode_steps=750,
    reward_threshold=500,
)

register(
    id='Maze10easy-ent-10-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    kwargs={"maze_type":10, "entropy_weight":10.0, 'difficulty':'easy'},
    max_episode_steps=1000,
    reward_threshold=500,
)


register(
    id='Maze10-ent-100-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    kwargs={"maze_type":10, "entropy_weight":100.0},
    max_episode_steps=750,
    reward_threshold=500,
)


register(
    id='Maze10easy-ent-100-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    kwargs={"maze_type":10, "entropy_weight":100.0, 'difficulty':'easy'},
    max_episode_steps=1000,
    reward_threshold=500,
)

register(
    id='Maze-entropy-10-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    kwargs={"reward_entropy": True, "entropy_weight": 10},
    max_episode_steps=500,
    reward_threshold=10,
)

register(
    id='Maze-entropy-100-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    kwargs={"reward_entropy": True, "entropy_weight": 100},
    max_episode_steps=500,
    reward_threshold=10,
)

register(
    id='Maze-entropy-only-v0',
    entry_point='brl_gym.wrapper_envs:BayesMazeEntropyEnv',
    max_episode_steps=500,
    reward_threshold=10,
)

register(
    id='Maze-entropy-only-no-reward-v0',
    entry_point='brl_gym.wrapper_envs:BayesMazeEntropyEnv',
    kwargs={'reward_entropy': False},
    max_episode_steps=500,
    reward_threshold=10,
)

register(
    id='Maze-entropy-hidden-v0',
    entry_point='brl_gym.wrapper_envs:BayesMazeEntropyEnv',
    kwargs={"reward_entropy": True, "observe_entropy":False},
    max_episode_steps=500,
    reward_threshold=10,
)

register(
    id='Maze-entropy-hidden-no-reward-v0',
    entry_point='brl_gym.wrapper_envs:BayesMazeEntropyEnv',
    kwargs={"reward_entropy": False, "observe_entropy": False},
    max_episode_steps=500,
    reward_threshold=10,
)

register(
    id='Maze10-entropy-hidden-ent-10-v0',
    entry_point='brl_gym.wrapper_envs:BayesMazeEntropyEnv',
    kwargs={
    "maze_type": 10,
    "reward_entropy": True,
    "entropy_weight":10.0,
    "observe_entropy": False},
    max_episode_steps=750,
    reward_threshold=500,
)

register(
    id='Maze10-entropy-hidden-ent-100-v0',
    entry_point='brl_gym.wrapper_envs:BayesMazeEntropyEnv',
    kwargs={
    "maze_type": 10,
    "reward_entropy": True,
    "entropy_weight":100.0,
    "observe_entropy": False},
    max_episode_steps=750,
    reward_threshold=500,
)

register(
    id='Maze10-entropy-hidden-noent-v0',
    entry_point='brl_gym.wrapper_envs:BayesMazeEntropyEnv',
    kwargs={
    "maze_type": 10,
    "reward_entropy": False,
    "entropy_weight":0.0,
    "observe_entropy": False},
    max_episode_steps=750,
    reward_threshold=500,
)

register(
    id='Maze10easy-entropy-hidden-noent-v0',
    entry_point='brl_gym.wrapper_envs:BayesMazeEntropyEnv',
    kwargs={
    "maze_type": 10,
    "reward_entropy": False,
    "entropy_weight":0.0,
    "observe_entropy": False,
    "difficulty": "easy"},
    max_episode_steps=1000,
    reward_threshold=500,
)

register(
    id='Maze10-entropy-only-ent-100-v0',
    entry_point='brl_gym.wrapper_envs:BayesMazeEntropyEnv',
    kwargs={
    "maze_type": 10,
    "reward_entropy": True,
    "entropy_weight":100.0,
    "observe_entropy": True},
    max_episode_steps=750,
    reward_threshold=500,
)

register(
    id='Maze10-entropy-only-noent-v0',
    entry_point='brl_gym.wrapper_envs:BayesMazeEntropyEnv',
    kwargs={
    "maze_type": 10,
    "reward_entropy": False,
    "entropy_weight": 0.0,
    "observe_entropy": True},
    max_episode_steps=750,
    reward_threshold=500,
)

register(
    id='MazeCont-noent-v0',
    entry_point='brl_gym.wrapper_envs:BayesMazeContinuousEnv',
    kwargs={"entropy_weight":0.0},
    max_episode_steps=750,
    reward_threshold=500
)

register(
    id='MazeCont-upmle-v0',
    entry_point='brl_gym.wrapper_envs:UPMLEMazeContEnv',
    kwargs={"entropy_weight":1.0,
        "reward_entropy":True},
    max_episode_steps=750,
    reward_threshold=500
)


register(
    id='MazeCont-upmle-noent-v0',
    entry_point='brl_gym.wrapper_envs:UPMLEMazeContEnv',
    kwargs={"entropy_weight":0.0,
        "reward_entropy":False},
    max_episode_steps=750,
    reward_threshold=500
)

register(
    id='Maze-no-entropy-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    kwargs={"reward_entropy": False},
    max_episode_steps=500,
    reward_threshold=10,
)


register(
    id='Maze10-no-entropy-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    kwargs={"reward_entropy": False, "maze_type": 10},
    max_episode_steps=750,
    reward_threshold=500,
)

register(
    id='Maze10easy-noent-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesMazeEnv',
    kwargs={"reward_entropy": False, "maze_type": 10, "difficulty": "easy"},
    max_episode_steps=1000,
    reward_threshold=500,
)

register(
    id='Maze-upmle-v0',
    entry_point='brl_gym.wrapper_envs:UPMLEMazeEnv',
    max_episode_steps=500,
    reward_threshold=10,
)

register(
    id='Maze10easy-upmle-v0',
    entry_point='brl_gym.wrapper_envs:UPMLEMazeEnv',
    kwargs={"maze_type":10, "entropy_weight":1.0, 'difficulty':'easy'},
    max_episode_steps=1000,
    reward_threshold=500,
)

register(
    id='Maze10-upmle-ent-100-v0',
    entry_point='brl_gym.wrapper_envs:UPMLEMazeEnv',
    kwargs={"maze_type": 10, "entropy_weight":100},
    max_episode_steps=750,
    reward_threshold=500,
)

register(
    id='Maze10easy-upmle-ent-100-v0',
    entry_point='brl_gym.wrapper_envs:UPMLEMazeEnv',
    kwargs={"maze_type":10, "entropy_weight":100.0, 'difficulty':'easy'},
    max_episode_steps=1000,
    reward_threshold=500,
)

register(
    id='Maze10-upmle-no-reward-v0',
    entry_point='brl_gym.wrapper_envs:UPMLEMazeEnv',
    kwargs={"reward_entropy": False, "maze_type": 10},
    max_episode_steps=750,
    reward_threshold=500,
)


register(
    id='Maze10easy-upmle-no-reward-v0',
    entry_point='brl_gym.wrapper_envs:UPMLEMazeEnv',
    kwargs={"maze_type":10, "reward_entropy": False,
    "entropy_weight":0.0, 'difficulty':'easy'},
    max_episode_steps=1000,
    reward_threshold=500,
)

register(
    id='Maze-upmle-no-reward-v0',
    entry_point='brl_gym.wrapper_envs:UPMLEMazeEnv',
    kwargs={"reward_entropy": False},
    max_episode_steps=500,
    reward_threshold=10,
)

register(
    id='Maze-upmle-no-entropy-v0',
    entry_point='brl_gym.wrapper_envs:UPMLEMazeEnvNoEntropyReward',
    max_episode_steps=500,
    reward_threshold=10,
)


register(
    id='Door-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesDoorsEnv',
    max_episode_steps=300,
    reward_threshold=100,
)

register(
    id='Door-LearnableBF-noent-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesDoorsEnv',
    max_episode_steps=300,
    reward_threshold=100,
    kwargs={
    "entropy_weight":0.0,
    "reward_entropy":False,
    "learnable_bf":True
    },
)


register(
    id='Doorslow-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesDoorsEnv',
    kwargs={"doors_slow":True},
    max_episode_steps=3000,
    reward_threshold=100,
)

register(
    id='Door-entropy-10-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesDoorsEnv',
    kwargs={"reward_entropy": True, "entropy_weight": 10},
    max_episode_steps=300,
    reward_threshold=100,
)

register(
    id='Door-entropy-100-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesDoorsEnv',
    kwargs={"reward_entropy": True, "entropy_weight": 100},
    max_episode_steps=300,
    reward_threshold=100,
)

register(
    id='Door-no-entropy-v0',
    entry_point='brl_gym.wrapper_envs:ExplicitBayesDoorsEnvNoEntropyReward',
    max_episode_steps=300,
    reward_threshold=100,
)

register(
    id='Door-upmle-v0',
    entry_point='brl_gym.wrapper_envs:UPMLEDoorsEnv',
    max_episode_steps=300,
    reward_threshold=100,
)

register(
    id='Door-upmle-no-entropy-v0',
    entry_point='brl_gym.wrapper_envs:UPMLEDoorsEnvNoEntropyReward',
    max_episode_steps=300,
    reward_threshold=100,
)

register(
    id='Door-entropy-only-v0',
    entry_point='brl_gym.wrapper_envs:BayesDoorsEntropyEnv',
    max_episode_steps=500,
    reward_threshold=10,
)

register(
    id='Door-entropy-hidden-v0',
    entry_point='brl_gym.wrapper_envs:BayesDoorsEntropyEnv',
    kwargs={"observe_entropy": False},
    max_episode_steps=500,
    reward_threshold=10,
)

register(
    id='Door-entropy-hidden-no-reward-v0',
    entry_point='brl_gym.wrapper_envs:BayesDoorsEntropyEnv',
    kwargs={"reward_entropy": False, "observe_entropy": False},
    max_episode_steps=500,
    reward_threshold=10,
)

register(
    id='Door-entropy-only-no-reward-v0',
    entry_point='brl_gym.wrapper_envs:BayesDoorsEntropyEnv',
    kwargs={'reward_entropy': False},
    max_episode_steps=500,
    reward_threshold=10,
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
