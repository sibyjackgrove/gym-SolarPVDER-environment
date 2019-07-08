
import gym

import matplotlib.pyplot as plt

import numpy as np

from gym_PVDER.agents.agent_utilities import Utilities

class QAgent(Utilities):
    
    # Exploration settings
    
    train_spec = {'LEARNING_RATE':0.1,
                  'DISCOUNT':0.95,
                  'EPISODES':200,
                  'SHOW_EVERY':10,
                   'STATS_EVERY':5}
    
    learn_spec = {'START_EPSILON_DECAYING': 1,
                  'END_EPSILON_DECAYING':train_spec['EPISODES']//2}
    
    # Exploration settings    
    explore_spec = {'EPSILON_DECAY': 0.98,
                    'MIN_EPSILON':0.001}
    
    epsilon = 1  # not a constant, qoing to be decayed    
    epsilon_decay_value = epsilon/(learn_spec['END_EPSILON_DECAYING'] - learn_spec['START_EPSILON_DECAYING'])
    
    def __init__(self, env):           
        
        assert isinstance(env,gym.wrappers.time_limit.TimeLimit),"Environment should be a Gym environment."
        
        self.env = env
        
        self.DISCRETE_OS_SIZE = len(self.env.observation_space.high)*[6]      #[20, 20]   
        print('Q table shape:',self.DISCRETE_OS_SIZE + [self.env.action_space.n])
        self.q_table = np.random.uniform(low=-2, high=-2, size=(self.DISCRETE_OS_SIZE + [self.env.action_space.n]))
        self.discrete_os_win_size = (self.env.observation_space.high - self.env.observation_space.low)/self.DISCRETE_OS_SIZE
        
        self.aggregate_episode_rewards = {'episode': [], 'average': [], 'max': [], 'min': []}
    
    def get_discrete_state(self,state):
        discrete_state = (state - self.env.observation_space.low)/self.discrete_os_win_size
        return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

    def train(self):
        
         # For stats
        episode_rewards = []
       
        
        print('Starting Q agent training for environment {}!'.format(self.env.name))
        
        for episode in range(self.train_spec['EPISODES']):
            episode_reward = 0
            discrete_state = self.get_discrete_state(self.env.reset())
            done = False

            #if episode % self.train_spec['SHOW_EVERY'] == 0:
            #    render = True
            #    print('Episode:',episode)
            #else:
            #    render = False

            while not done:

                if np.random.random() > self.epsilon:
                    # Get action from Q table
                    action = np.argmax(self.q_table[discrete_state])
                else:
                    # Get random action
                    action = np.random.randint(0, self.env.action_space.n)

                new_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                new_discrete_state = self.get_discrete_state(new_state)

                if episode % self.train_spec['SHOW_EVERY'] == 0 and done:
                    print('Environment render @ Episode:{},Steps:{}'.format(episode+1,self.env.steps))
                    self.env.render(mode='vector')
                #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                # If simulation did not end yet after last step - update Q table
                if not done:

                    # Maximum possible Q value in next step (for new state)
                    max_future_q = np.max(self.q_table[new_discrete_state])

                    # Current Q value (for current state and performed action)
                    current_q = self.q_table[discrete_state + (action,)]

                    # And here's our equation for a new Q value for current state and action
                    new_q = (1 - self.train_spec['LEARNING_RATE']) * current_q + self.train_spec['LEARNING_RATE'] * (reward + self.train_spec['DISCOUNT'] * max_future_q)

                    # Update Q table with new Q value
                    self.q_table[discrete_state + (action,)] = new_q


                # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
                else:
                    self.q_table[discrete_state + (action,)] = reward
                #elif new_state[0] >= self.env.goal_position:
                    #q_table[discrete_state + (action,)] = reward
                    self.q_table[discrete_state + (action,)] = 0

                discrete_state = new_discrete_state

            # Decaying is being done every episode if episode number is within decaying range
            #if self.learn_spec['END_EPSILON_DECAYING'] >= episode >= self.learn_spec['START_EPSILON_DECAYING']:
            #    self.epsilon -= self.epsilon_decay_value
            self.update_epsilon()
                
            episode_rewards.append(episode_reward)
            if not episode % self.train_spec['STATS_EVERY']:
                print('Storing rewards @ Episode:{},Steps:{}'.format(episode+1,self.env.steps))
                average_reward = sum(episode_rewards[-self.train_spec['STATS_EVERY']:])/self.train_spec['STATS_EVERY']
                max_reward = max(episode_rewards[-self.train_spec['STATS_EVERY']:])
                min_reward = min(episode_rewards[-self.train_spec['STATS_EVERY']:])
                
                self.collect_aggregate_rewards(episode,average_reward,min_reward,max_reward)

        print('Average reward over full training:{}'.format(sum(episode_rewards)/len(episode_rewards)))
        print('Maximum reward over full training:{}'.format(max(episode_rewards)))
        print('Minimum reward over full training:{}'.format(min(episode_rewards)))
        
        self.env.close()      
        
    def agent_action(self,observation):
        """Get action from agent."""
        
        discrete_state = self.get_discrete_state(observation)
        
        return np.argmax(self.q_table[discrete_state])
