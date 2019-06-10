import pytest
import numpy as np

import gym
import gym_PVDER
from gym_PVDER.envs import PVDER_env

#@pytest.mark.parametrize("spec", spec_list)

# Check if environment can be created
def test_make():
    env = gym.make('PVDER-v0')
    assert env.spec.id == 'PVDER-v0'
    assert isinstance(env.unwrapped, PVDER_env.PVDER)

# Check environment specs
def test_env():
    # Capture warnings
    with pytest.warns(None) as warnings:
        gym_spec = gym.spec('PVDER-v0')
        env = gym_spec.make()

    # Check that dtype is explicitly declared for gym.Box spaces
    for warning_msg in warnings:
        assert not 'autodetected dtype' in str(warning_msg.message)

    ob_space = env.observation_space
    act_space = env.action_space
    ob = env.reset()
    assert ob_space.contains(ob), 'Reset observation: {!r} not in space'.format(ob)
    a = act_space.sample()
    observation, reward, done, _info = env.step(a)
    assert ob_space.contains(observation), 'Step observation: {!r} not in space'.format(observation)
    assert np.isscalar(reward), "{} is not a scalar for {}".format(reward, env)
    assert isinstance(done, bool), "Expected {} to be a boolean".format(done)

    for mode in env.metadata.get('render.modes', []):
        env.render(mode=mode)
    
    env.close()

# Run a longer rollout
def test_random_rollout():
    for env in [gym.make('PVDER-v0')]:
        agent = lambda ob: env.action_space.sample()
        ob = env.reset()
        for _ in range(10):
            assert env.observation_space.contains(ob)
            a = agent(ob)
            assert env.action_space.contains(a)
            (ob, _reward, done, _info) = env.step(a)
            if done: break
        env.close()
