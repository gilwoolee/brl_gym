from brl_gym.wrapper_envs.wrapper_maze import Expert, ExplicitBayesMazeEnv
import numpy as np
if __name__ == "__main__":

    expert = Expert(nenv=1, use_vf=True)

    all_rewards = []
    for _ in range(500):
        env = ExplicitBayesMazeEnv(reward_entropy=False)
        o = env.reset()

        rewards = []
        for t in range(500):
            action = expert.action(
                np.concatenate([o['obs'], o['zbel']]).reshape(1, -1)).ravel()

            # print(action)
            # if t < 100:
            action[2] = action[2] + np.random.normal()
            o, r, d, _ = env.step(action)
            # env.render()
            # print(np.around(o['zbel'], 2))
            rewards += [r]
            if d:
                break

        undiscounted_sum = np.sum(rewards)
        all_rewards += [undiscounted_sum]
        print('undiscounted sum', undiscounted_sum)

    print("stat", np.mean(all_rewards), np.std(all_rewards) / np.sqrt(len(all_rewards)))
    # 229.978 17.471892485704004
    import IPython; IPython.embed();

