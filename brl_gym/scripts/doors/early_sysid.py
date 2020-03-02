from brl_gym.wrapper_envs.wrapper_doors import Expert, ExplicitBayesDoorsEnv
import numpy as np
if __name__ == "__main__":

    expert = Expert()

    all_rewards = []
    num_sensing = []
    for _ in range(100):
        env = ExplicitBayesDoorsEnv(reward_entropy=False)
        o = env.reset()

        rewards = []
        sensing = 0
        for t in range(300):
            action = expert.action(
                np.concatenate([o['obs'], o['zbel']]).reshape(1, -1)).ravel()

            # print(action)
            # if t < 100:
            action[2] = action[2] + np.random.normal()*0.1
            o, r, d, _ = env.step(action)
            # env.render()
            # print(np.around(o['zbel'], 2))
            rewards += [r]
            if action[2] > 0.0:
                sensing += 1

            if d:
                print(t)
                break

        undiscounted_sum = np.sum(rewards)
        num_sensing += [sensing]
        all_rewards += [undiscounted_sum]
        print('undiscounted sum', undiscounted_sum, "sensing", sensing)

    print("stat", np.mean(all_rewards), np.std(all_rewards) / np.sqrt(len(all_rewards)))
    print("sensing", np.mean(num_sensing), np.std(num_sensing) / np.sqrt(len(num_sensing)))
     # -108.5296 16.521
    import IPython; IPython.embed();

