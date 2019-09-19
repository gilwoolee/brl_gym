from brl_gym.envs.mujoco.motion_planner.VectorFieldGenerator import VectorField

for i in range(10):
	vf = VectorField(make_new=True, maze_type=10, target=i)
