import julia
import numpy as np
import os.path as osp
import gym
from brl_gym.envs.mujoco import box_pusher
env = box_pusher.BoxPusher()

rlopt = "/home/gilwoo/School_Workspace/rlopt"
j = julia.Julia()

j.include(osp.join(rlopt, "_init.jl"))
j.include(osp.join(rlopt, "src/pg/Baseline.jl"))
j.include(osp.join(rlopt, "src/ExpSim.jl"))
polo = "/tmp/pusher_polo_opt_1"
baseline = j.Baseline.loadbaseline(osp.join(polo, "baseline.jld2"))
datafile = j.ExpSim.load(osp.join(polo, "data.jld2"))

# Replay the datafile
state = np.squeeze(datafile["state"]).transpose()
ctrl = np.squeeze(datafile["ctrl"]).transpose()
obs = np.squeeze(datafile["obs"]).transpose()

# a = [x for x in a.split(";")]
# data = []
# for x in a:
#     data += [[float(y) for y in x.split(" ") if y != " " and y != ""]]

# data = np.array(data)
# value = j.Baseline.predict(baseline, data.tolist())
# print("Value", value)

o = env.reset()

# env.set_state_ctrl(state[1,:35], state[1,35:], ctrl[1])
#o, r, d, _ = env.step(ctrl[0])
# new_state = env.sim.get_state()
# import IPython; IPython.embed(); import sys; sys.exit(0)
#copy_env = humanoid_pushing.HumanoidPushingEnv()
#copy_env.reset()

print(state.shape)
states = []
observations = []
rewards = []
values = []
for i in range(state.shape[0]):
    env.set_state(state[i,:5], state[i,5:])
    # states += [(env.sim.get_state().qpos, env.sim.get_state().qvel)]
    o, r, d, _ = env.step(ctrl[i])
    # observations += [o]
    # rewards += [r]
    values += [j.Baseline.predict(baseline, o.reshape(-1,1).tolist())]
    env.render()


import IPython; IPython.embed()
