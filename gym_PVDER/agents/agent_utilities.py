"""Utilities for agent"""

import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

class Utilities():
    """Miscelleneous utilities."""
    
    valid_stats = ['episodic_reward','average_reward','min_reward','max_reward']
    
    
    def initialize_aggregate_rewards(self,reward_stats =  ['average_reward','min_reward','max_reward']):
        """Initialize dictionary collecting rewards statistics."""
        
        self.aggregate_episode_rewards = {}
        self.aggregate_episode_rewards.update({'episode': []})
        
        for stat_type in reward_stats:
            if stat_type in self.valid_stats:
                self.aggregate_episode_rewards.update({stat_type: []})            
            else:
                ValueError(f'{stat_type} is invalid!')        
                
        self._reward_stats = reward_stats
    
    def collect_aggregate_rewards(self, episode, **reward_stats): #episode,average_reward,min_reward,max_reward
        """Collect rewards statistics."""
        
        print('Storing rewards @ Episode:{},Steps:{}'.format(episode,self.env.steps))       
       
        self.aggregate_episode_rewards['episode'].append(episode) 
        
        for stat_type,value in reward_stats.items():
            if stat_type in self._reward_stats:
                self.aggregate_episode_rewards[stat_type].append(value)
                print(stat_type,':',value) 
            else:
                ValueError(f'{stat_type} is invalid!')
            
            #self.aggregate_episode_rewards['episode'].append(episode)
            #self.aggregate_episode_rewards['average'].append(average_reward)
            #self.aggregate_episode_rewards['min'].append(min_reward)
            #self.aggregate_episode_rewards['max'].append(max_reward)        

        #print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {self.epsilon:.3f}')        
    
    def show_plots(self):
        """Show plots."""
        
        for stat_type in self._reward_stats:
            if stat_type in self.valid_stats:
                plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards[stat_type], label=stat_type)
            else:
                ValueError(f'{stat_type} is invalid!') 
        
        #plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards['average'], label="average rewards")
        #plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards['max'], label="max rewards")
        #plt.plot(self.aggregate_episode_rewards['episode'], self.aggregate_episode_rewards['min'], label="min rewards")
        plt.legend(loc=4)
        plt.show()        
        
    def evaluate(self,n_episodes=1):
        """Evaluate agent."""
        
        for i in range(n_episodes):
            observation = self.env.reset()
            done = False
            while not done:
                action = self.agent_action(observation)
                print('Step:{},Action:{}'.format(self.env.steps,action))
                observation, reward, done, _ = self.env.step(action)
                self.env.render(mode='vector')        
            self.env.render(mode='human')
            
    def update_epsilon(self):
        """Decay epsilon every episode."""
        
        if self.epsilon > self.explore_spec['MIN_EPSILON']:
            self.epsilon *= self.explore_spec['EPSILON_DECAY']
            self.epsilon = max(self.explore_spec['MIN_EPSILON'], self.epsilon)   

class ModifiedTensorBoard(TensorBoard):
    """ Own Tensorboard class"""
    
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        #self.writer = tf.summary.FileWriter(self.log_dir)
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):#**stats
                
        #print(self.step,stats)
        self._write_custom_summaries(self.step,stats)
