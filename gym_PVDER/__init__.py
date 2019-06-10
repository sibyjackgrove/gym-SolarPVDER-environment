from gym.envs.registration import register

register(
    id='PVDER-v0',
    entry_point='gym_PVDER.envs:PVDER',
    kwargs={'n_sim_time_steps_per_env_step': 30,
            'max_sim_time' : 30.0,
            'DISCRETE_REWARD' : True,
            'goals_list':['voltage_regulation']},
    max_episode_steps=500,)
    #tags={'wrapper_config.TimeLimit.max_episode_steps': 10}
    