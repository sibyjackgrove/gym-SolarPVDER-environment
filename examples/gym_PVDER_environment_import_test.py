

# ## Import the modules and check if PVDER environment can be created

# In[ ]:


import gym
import gym_PVDER


# In[ ]:


env = gym.make('PVDER-v0')


# ## Create a random agent and iteract with the environment

# In[ ]:


episode_count = 1
for i in range(episode_count):
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  #Sample actions from the environment's discrete action space
        print('Action:',action)
        observation, reward, done, _ = env.step(action)
        env.render()
        


# In[ ]:


env.env.sim.PV_model.Q_ref


# In[ ]:


env.env.steps

