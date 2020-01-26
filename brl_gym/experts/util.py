
def get_expert(env_name, num_env, use_mle, **kwargs):

    if "Maze" in env_name:
        from brl_gym.experts.maze.expert import MazeExpert
        if env_name.startswith("Maze10"):
            return MazeExpert(nenv=num_env, mle=use_mle, maze_type=10)
        elif args.env.startswith("MazeCont"):
            from brl_gym.wrapper_envs.wrapper_maze_continuous import Expert
            return Expert()
        else:
            return MazeExpert(nenv=num_env, mle=use_mle, maze_type=4)

    elif "Door".lower() in env_name.lower():
        from brl_gym.experts.doors.expert import DoorsExpert
        return DoorsExpert(mle=use_mle)

    elif 'LightDarkHard'.lower() in env_name.lower():
        from brl_gym.experts.lightdarkhard.expert import LightDarkHardExpert
        return LightDarkHardExpert()

    elif 'LightDark'.lower() in env_name.lower():
        from brl_gym.experts.lightdark.expert import LightDarkExpert
        return LightDarkExpert()

    elif 'ContinuousCartPole'.lower() in env_name.lower():
        from brl_gym.experts.classic_control.continuous_cartpole_expert import ContinuousCartPoleExpert
        return ContinuousCartPoleExpert()

    elif "WamFindObj".lower() in env_name.lower():
        from brl_gym.experts.mujoco.wam_find_obj_expert import WamFindObjExpert
        return WamFindObjExpert()

    elif "crosswalk".lower() in env_name.lower():
        from brl_gym.experts.crosswalk.expert import CrossWalkExpert
        return CrossWalkExpert()
