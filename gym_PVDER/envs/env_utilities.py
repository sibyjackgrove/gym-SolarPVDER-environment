"""Utilities for environment."""

import pprint

import matplotlib.pyplot as plt

class Utilities():
    """Miscelleneous utilities."""
    
    pp = pprint.PrettyPrinter(indent=4)
    
    def initialize_stats(self):
        """Initialize actions statistics."""
        
        self._action_stats = {}
        
        for i in range(self.action_space.n):
            self._action_stats[i] = 0    
        
        self._reward_stats = {'step':[],'reward':[]}
        self._time_stats = {'step':[],'step_time':[]}
        
        self.logger.debug('{}:Stats initialized for environment.'.format(self.name))
    
    def update_action_stats(self,action):
        """Update actions statistics."""
        
        self._action_stats[action] = self._action_stats[action] + 1
        
        self.logger.debug('{}:Counter for action {} incremented.'.format(self.name,action))
    
    def update_reward_stats(self):
        """Collect rewards every step."""
        
        self._reward_stats['step'].append(self._steps)
        self._reward_stats['reward'].append(self._reward)        
        
        self.logger.debug('{}:Time stats updated.'.format(self.name))
    
    def update_time_stats(self):
        """Collect step times in episode."""
        
        self._time_stats['step'].append(self._steps)
        self._time_stats['step_time'].append(self._step_time)        
        
        self.logger.debug('{}:Time stats updated.'.format(self.name))
    
    @property
    def steps(self):
        """Return steps counter value (since it is private attribute)."""
        
        return self._steps
    
    def show_step_time(self):
        """Show time take by a single step."""
        
        print('Time for step:{:.3f}'.format(self._step_time))
        
    def show_action_stats(self,SHOW_PLOT=False):
        """Show actions count for episode."""
        
        self.pp.pprint(self._action_stats)
        
        if SHOW_PLOT:
            action_list = list(self._action_stats.keys()) 
            action_count = list(self._action_stats.values())

            x_pos = [i for i, _ in enumerate(action_list)]

            plt.bar(x_pos, action_count, color='green')
            plt.xlabel("Actions")
            plt.ylabel("Count")
            plt.title("Action totals after every episode.")
            plt.xticks(x_pos, action_list)
            plt.show()    
      
    def show_reward_stats(self,SHOW_PLOT=False):
        """Show reward per step for episode."""
        
        if SHOW_PLOT:
            
            plt.plot(self._reward_stats['step'], self._reward_stats['reward'], color='red')
            plt.xlabel("Steps")
            plt.ylabel("Reward")
            plt.title("Reward per step")
            plt.show() 
        else:
            self.pp.pprint(self._reward_stats)
    
    def show_time_stats(self,SHOW_PLOT=False):
        """Show time per step for episode."""
        
        if SHOW_PLOT:
            
            plt.plot(self._time_stats['step'], self._time_stats['step_time'], color='blue')
            plt.xlabel("Steps")
            plt.ylabel("Time (s)")
            plt.title("Time per step")
            plt.show()
            
        else:
            self.pp.pprint(self._time_stats)