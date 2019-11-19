from brl_gym.wrapper_envs.wrapper_maze import Expert, ExplicitBayesMazeEnv
import numpy as np
if __name__ == "__main__":

    expert = Expert(nenv=1, use_vf=True, maze_type=10)

    all_rewards = []
    all_top_belief = []
    completed = 0
    for i in range(300):
        env = ExplicitBayesMazeEnv(maze_type=10, difficulty='easy', entropy_weight=0.0)
        o = env.reset()

        rewards = []
        for t in range(750):
            action = expert.action(
                np.concatenate([o['obs'], o['zbel']]).reshape(1, -1)).ravel()

            # print(action)
            #if t < 150:
            #    action[2] = action[2] + 1
            action[2] = np.random.normal()

            o, r, d, _ = env.step(action)
            # env.render()
            rewards += [r]
            if t == 150:
                print(np.around(o['zbel'],2))
                all_top_belief += [np.max(o['zbel'])]

            if d:
                completed += 1
                print(i, "complete {:3f}".format(completed / (i + 1)))
                break

        undiscounted_sum = np.sum(rewards)
        all_rewards += [undiscounted_sum]
        print('undiscounted sum', undiscounted_sum)

    print("stat", np.mean(all_rewards), np.std(all_rewards) / np.sqrt(len(all_rewards)))
    print("average top belief", np.mean(all_top_belief), np.std(all_top_belief) / np.sqrt(len(all_rewards)))
    # stat 512.4810662596318 14.01378592003006
    import IPython; IPython.embed();
