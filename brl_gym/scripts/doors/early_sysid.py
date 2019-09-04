from brl_gym.wrapper_envs.wrapper_doors import Expert, ExplicitBayesDoorsEnv
import numpy as np
if __name__ == "__main__":

    expert = Expert()

    all_rewards = []
    for _ in range(1):
        env = ExplicitBayesDoorsEnv(reward_entropy=False)
        o = env.reset()

        rewards = []
        for t in range(300):
            action = expert.action(
                np.concatenate([o['obs'], o['zbel']]).reshape(1, -1)).ravel()

            # print(action)
            if t < 100:
                action[2] = action[2] + np.random.normal()*0.1
            o, r, d, _ = env.step(action)
            env.render()
            # print(np.around(o['zbel'], 2))
            rewards += [r]
            if d:
                break

        undiscounted_sum = np.sum(rewards)
        all_rewards += [undiscounted_sum]
        print('undiscounted sum', undiscounted_sum)

    print("stat", np.mean(all_rewards), np.std(all_rewards) / np.sqrt(len(all_rewards)))
     # -108.5296 16.521
    import IPython; IPython.embed();

