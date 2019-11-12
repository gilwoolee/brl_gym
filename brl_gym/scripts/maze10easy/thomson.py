from brl_gym.wrapper_envs.wrapper_maze import Expert, ExplicitBayesMazeEnv, simple_expert_actor
from brl_gym.envs.mujoco.maze10 import GOAL_POSE

import numpy as np
if __name__ == "__main__":

    expert = Expert(nenv=1, use_vf=True, maze_type=10)

    all_rewards = []
    completed = 0
    for i in range(300):
        env = ExplicitBayesMazeEnv(maze_type=10, difficulty='easy', entropy_weight=0.0)
        o = env.reset()

        rewards = []
        for t in range(750):
            bel = o['zbel']
            state = o['obs'][:2]
            idx = np.random.choice(np.arange(10), p=bel)

            chosen_expert = expert.mps[idx]
            action = simple_expert_actor(chosen_expert, state, GOAL_POSE[idx])
            action = np.concatenate([action, np.array([0])])

            action[2] = action[2] + np.random.normal()*0.1
            action += np.random.normal(size=3)*0.1
            o, r, d, _ = env.step(action)
            if r <= -50:
                r = 0 # temporarily remove the hardship
            # env.render()
            # print(np.around(o['zbel'], 2))
            rewards += [r]
            if d:
                print(i, "complete")
                completed += 1
                break

        undiscounted_sum = np.sum(rewards)
        all_rewards += [undiscounted_sum]
        print('undiscounted sum', undiscounted_sum)

    print("stat", np.mean(all_rewards), np.std(all_rewards) / np.sqrt(len(all_rewards)))
    print("Completed", completed, "/", 500)
    # stat 512.4810662596318 14.01378592003006
    import IPython; IPython.embed();
