from brl_gym.experts.maze.expert import MazeExpert
from brl_gym.experts.lightdark.expert import LightDarKExpert
from brl_gym.experts.doors.expert import DoorsExpert

def get_expert(env_name, num_env, use_mle, **kwargs):

    if "Maze" in env_name:
        if env_name.startswith("Maze10"):
            return MazeExpert(nenv=num_env, mle=use_mle, maze_type=10)
        elif args.env.startswith("MazeCont"):
            from brl_gym.wrapper_envs.wrapper_maze_continuous import Expert
            return Expert()
        else:
            return MazeExpert(nenv=num_env, mle=use_mle, maze_type=4)

    elif "Doors" in env_name:
        return DoorsExpert(mle=use_mle)

    elif 'LightDark' in env_name:
        return LightDarKExpert()

