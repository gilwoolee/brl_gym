from brl_gym.envs.crosswalk_vel import CrossWalkVelEnv
import numpy as np

env = CrossWalkVelEnv()
env.reset()
goals = env.goals
peds = env.pedestrians
pose = env.pose
ped_speeds = env.pedestrian_speeds

print("Car 37, 38, 35")
print("Peds      :\n", np.around(peds,1))
print("Ped speeds:\n", np.around(ped_speeds,2))
print("Goals     :\n", np.around(goals,1))
print("Pose      :\n", np.around(pose,1))
print("Angle     :\n", np.around(np.rad2deg(pose[2]),2))

for ps, goal in zip(ped_speeds, goals):
	if goal[0] == 3.5:
		goal[0] = 3.2
	if goal[0] == 0.0:
		goal[0] = 0.3
	print("roslaunch mushr_control runner_script.launch car_name:=$CAR_NAME wait_for_signal:=false desired_speed:={:.2f} desired_x:={:.2f} desired_y:={:.2f} local:=false".format(ps, goal[0], goal[1]))

