from gym.envs.registration import register

register(
    id='PVDER-v0',
    entry_point='gym_PVDER.envs:PVDER',
    max_episode_steps=900,)