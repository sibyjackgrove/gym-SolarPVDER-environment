from gym.envs.registration import register

register(
    id='PVDER-v0',
    entry_point='gym_PVDER.envs:PVDER',
    kwargs={'n_sim_time_steps_per_environment_step': 30,
            'max_simulation_time' : 20.0,
            'DISCRETE_REWARD' : False,
            'goal_list':['voltage_regulation']},
    max_episode_steps=500,)
    #tags={'wrapper_config.TimeLimit.max_episode_steps': 10}    