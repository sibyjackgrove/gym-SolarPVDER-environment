import pytest
import numpy as np

import gym
import gym_DER
from gym_DER.envs import PVDER_env

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
        assert 'autodetected dtype' not in str(warning_msg.message)

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
            if done:
                break
        env.close()

# Test if environment is giving discrete reward
def test_discrete_reward():
    for env in [gym.make('PVDER-v0',DISCRETE_REWARD=True,goals_list=['voltage_regulation'])]:
        agent = lambda ob: env.action_space.sample()
        ob = env.reset()
        for _ in range(10):
            assert env.observation_space.contains(ob)
            a = agent(ob)
            assert env.action_space.contains(a)
            (ob, _reward, done, _info) = env.step(a)
            assert isinstance(_reward,int),'Reward should be discrete if DISCRETE_REWARD is True'
            if done:
                break
        env.close()

# Test if environment is giving continuous reward
def test_continuous_reward():
    for env in [gym.make('PVDER-v0',DISCRETE_REWARD=False,goals_list=['voltage_regulation'])]:
        agent = lambda ob: env.action_space.sample()
        ob = env.reset()
        for _ in range(10):
            assert env.observation_space.contains(ob)
            a = agent(ob)
            assert env.action_space.contains(a)
            (ob, _reward, done, _info) = env.step(a)
            assert isinstance(_reward,float),'Reward should be float if DISCRETE_REWARD is False'
            if done:
                break
        env.close()
        
# Test if arguments are working
def test_time_steps():
    
    _max_sim_time = 25.0
    _n_sim_time_steps_per_env_step = 10
    
    for env in [gym.make('PVDER-v0',n_sim_time_steps_per_env_step =_n_sim_time_steps_per_env_step,max_sim_time =_max_sim_time)]:
        agent = lambda ob: env.action_space.sample()
        ob = env.reset()
        
        assert env.n_sim_time_steps_per_env_step == _n_sim_time_steps_per_env_step, 'Environment n_sim_time_steps_per_env_step is different from the value specifed by user!'
        assert env.max_sim_time == _max_sim_time, 'Environment max_sim_time is different from value specified by user!'
        
        done = False
        steps = 0 
        while not done:
            assert env.observation_space.contains(ob)
            a = agent(ob)
            assert env.action_space.contains(a)
            (ob, _reward, done, _info) = env.step(a)
            steps = steps + 1
            
        assert round(env.steps*env.n_sim_time_steps_per_env_step*env.sim.tInc,6) == round(env.max_sim_time,6), 'Error in steps check!'
        assert round(ob[-1]*env.max_sim_time,6) == round(env.max_sim_time,6), 'Last time instant should be equal to user specified max_sim_time!'
            
        env.close()

# Test if environment events can be updated
def test_update_env_events():
    
    new_spec ={'voltage': {'min':0.95}}
        
    for env in [gym.make('PVDER-v0')]:
        env.update_env_events(event_spec_list=[new_spec])
        agent = lambda ob: env.action_space.sample()
        ob = env.reset()
        
        assert env.env_events_spec['voltage']['min'] == new_spec['voltage']['min'], 'Environment events spec has not been modified!'
        assert env.sim.simulation_events._events_spec['voltage']['min'] == new_spec['voltage']['min'], 'Simulation object events spec has not been modified!'
        
        for _ in range(10):
            assert env.observation_space.contains(ob)
            a = agent(ob)
            assert env.action_space.contains(a)
            (ob, _reward, done, _info) = env.step(a)
            if done:
                break
        env.close()
