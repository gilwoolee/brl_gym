from brl_gym.wrapper_envs.wrapper_maze import Expert, ExplicitBayesMazeEnv
import numpy as np
if __name__ == "__main__":

    expert = Expert(nenv=1)

    all_rewards = []
    for _ in range(1):
        env = ExplicitBayesMazeEnv()
        o = env.reset()
        
        rewards = []
        for t in range(500):
            action = expert.action(
                np.concatenate([o['obs'], o['zbel']]).reshape(1, -1)).ravel()

            # print(action)
            if t < 100:
                action[2] = action[2] + np.random.normal()*0.1
            o, r, d, _ = env.step(action)
            env.render()
            # print(o['zbel'])
            rewards += [r]
            if d:
                break

        undiscounted_sum = np.sum(rewards)
        all_rewards += [undiscounted_sum]
        print('undiscounted sum', undiscounted_sum)

    print("stat", np.mean(all_rewards), np.std(all_rewards) / np.sqrt(len(all_rewards)))
    # stat 512.4810662596318 14.01378592003006
    import IPython; IPython.embed();
