from brl_gym.envs.crosswalk_vel import CrossWalkVelEnv
import numpy as np

env = CrossWalkVelEnv()
env.reset()
goals = env.goals
peds = env.pedestrians
pose = env.pose
ped_speeds = env.pedestrian_speeds

print("Peds      :\n", np.around(peds,1))
print("Ped speeds:\n", np.around(ped_speeds,2))
print("Goals     :\n", np.around(goals,1))
print("Pose      :\n", np.around(pose,1))
print("Angle     :\n", np.around(np.rad2deg(pose[2]),2))
