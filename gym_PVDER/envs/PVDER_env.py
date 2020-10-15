import gym
from gym import error, spaces, utils
from gym.utils import seeding

import sys
import os
import logging
import importlib.util
import pprint
import time

import matplotlib.pyplot as plt

import numpy as np
import random

# For illustrative purposes.
package_name = 'pvder'

spec = importlib.util.find_spec(package_name)
if spec is None:
    print(package_name +" is not installed! Please install the module from https://github.com/sibyjackgrove/SolarPV-DER-simulation-utility")

#np.set_printoptions(precision=2)  #for setting number of decimal places when printing numpy arrays

from pvder.DER_components_single_phase  import SolarPVDERSinglePhase  #Python 3 need full path to find module
from pvder.DER_components_three_phase  import SolarPVDERThreePhase
from pvder.DER_wrapper import DERModel
from pvder.grid_components import Grid
from pvder.simulation_events import SimulationEvents
from pvder.simulation_utilities import SimulationResults
from pvder.dynamic_simulation import DynamicSimulation
from pvder.utility_classes import Logging

from pvder import utility_functions

from gym_PVDER.envs.env_utilities import Utilities

class PVDER(gym.Env,Logging,Utilities):
    
    count = 0
    metadata = {'render.modes': ['vector','human']}
    
    observed_quantities = ['iaR','iaI',
                           'vaR','vaI',
                           'P_PCC','Q_PCC',
                           'Vdc','Ppv',
                           'Vdc_ref','Q_ref',
                           'tStart'] #,'maR','maI'
    action_space = spaces.Discrete(5)
    observation_space = spaces.Box(low=-10, high=10, shape=(len(observed_quantities),),dtype=np.float32)
    
    #objective_dict = {'Q_ref_+':{'Q_ref':0.1},'Q_ref_-':{'Q_ref':-0.1}}
    baseDir=os.path.dirname(os.path.abspath(__file__))
    
    env_model_spec = {'model_1':{'DERModelType':'SinglePhase','derId':'10','configFile':baseDir[:-15]+'/config_der.json'},
                      'model_2':{'DERModelType':'ThreePhaseUnbalanced','derId':'50','configFile':baseDir[:-15]+'/config_der.json'}
                     }
    
    env_events_spec = {'insolation':{'t_events_start':1.0,'t_events_stop':39.0,'t_events_step':1.0,'min':85.0,'max':100.0,'ENABLE':False},
                       'voltage':{'t_events_start':1.0,'t_events_stop':39.0,'t_events_step':1.0,'min':0.98,'max':1.02,'ENABLE':True}} 
    
    env_sim_spec = {'sim_time_step':(1/60),
                    'n_sim_time_steps_per_env_step':{'default':60,'min':1},
                    'min_sim_time':1.0}    
    
    env_reward_spec = {'reward_list':{'default':['voltage_error'],
                                      'valid': ['voltage_error','power_error','Q_error','Vdc_error']},
                       'reference_values':{'P_ref':45.4e3,'Q_ref':5.5e3},
                       'DISCRETE_REWARD':True
                      } #List with agent goals. 
    
    env_action_spec = {'action_list':{'default':['Q_control'],
                                      'valid': ['Q_control','Vdc_control']},
                       'delQref':25,'delVdcref':0.02} # delta per sim time step
    
    
    env_goal_spec = {'voltage_regulation':{'reward':{'required':['voltage_error'],
                                                     'optional': ['Q_error','Vdc_error']},
                                           'action':{'required':['Q_control'],
                                                     'optional': ['Vdc_control']}
                                          },
                     'power_regulation':{'reward':{'required':['power_error'],
                                                   'optional': ['Vdc_error']},
                                         'action':{'required':['Vdc_control'],
                                                   'optional': []}
                                         },
                     'Q_regulation':{'reward':{'required':['Q_error'],
                                               'optional': ['Vdc_error']},
                                     'action':{'required':['Q_control'],
                                               'optional': []}
                                         }
                    }
                     
    default_goal = ['voltage_regulation']
    _sim_time_per_env_step = env_sim_spec['sim_time_step']*env_sim_spec['n_sim_time_steps_per_env_step']['default']
    
    _reward_list = env_reward_spec['reward_list']['default']
    _action_list = env_action_spec['action_list']['default']
    
    _delQref = env_action_spec['delQref']*env_sim_spec['n_sim_time_steps_per_env_step']['default']
    _delVdcref = env_action_spec['delVdcref']*env_sim_spec['n_sim_time_steps_per_env_step']['default']
    
    def __init__(self, goals_list,n_sim_time_steps_per_env_step, max_sim_time,DISCRETE_REWARD, 
                 verbosity='INFO'):   
        """
        Setup PV-DER environment variables.
        
        Args:
           n_sim_time_steps_per_env_step (int): Scalar specifiying simulation time steps per environment step.           
           max_sim_time (float): Scalar specifiying maximum simulation time in seconds for simulation within the environment.
           goals_list (list): List of strings specifying agent goals.
           verbosity (string): Specify logging levels - ('DEBUG','INFO','WARNING').
        """
        
        #Increment count to keep track of number of gym-PVDER environment instances
        PVDER.count = PVDER.count+1
        self.name = 'GymDER_'+ str(self.count)
        self.initialize_logger(logging_level=verbosity) #Set logging level - {DEBUG,INFO,WARNING,ERROR} 
        
        self.n_sim_time_steps_per_env_step = n_sim_time_steps_per_env_step
        self.max_sim_time_user = max_sim_time
        
        self.goals_list = goals_list
        self.DISCRETE_REWARD = DISCRETE_REWARD
        
        self.verbosity = verbosity #Set logging level - {DEBUG,INFO,WARNING,ERROR}
        
        self.initialize_environment_variables()
        
        
        print('{}:Environment created!'.format(self.name))
    
    def __del__(self):
        # body of destructor
        print("Destructor called - {} deleted!".format(self.name))
    
    def step(self, action):
        
        time_start = time.time()
        
        if self._steps == 0:
            self.logger.info('{}:New episode started with tStart = {} s'.format(self.name,self.sim.tStart))
        
        if self.done and self.sim.tStop >= self.max_sim_time:
            self.logger.warning('{}:Simulation completed - Reset environment to start new simulation!'.format(self.name))
        
        elif self.done and self.CONVERGENCE_FAILURE:
            self.logger.warning('{}:Simulation stopped due to converge failure at {} s - Reset environment to start new simulation!'.format(self.name,self.sim.tStop))
        
        elif self.done and self.RUNTIME_ERROR:
            self.logger.warning('{}:Simulation stopped due to run time error failure at {} s - Check code for error in logic!'.format(self.name,self.sim.tStop))
        
        elif not self.done:
            self.sim.tStop = self.sim.tStart + self._sim_time_per_env_step #+ self.sim.tInc #
            self._steps = self._steps + 1
            #t = self.sim.t_calc()
            t,dt = np.linspace(self.sim.tStart, self.sim.tStop,self.n_sim_time_steps_per_env_step+1,retstep=True)
            #print(t,dt)
            assert round(dt,6) == round(self.sim.tInc,6), 'Generated step size should be equal to specified step size.'            

            self.action_calc(action)
            ConvergeFlag = False
            try:
                #solution,info,ConvergeFlag = self.sim.call_ODE_solver(self.sim.ODE_model,self.sim.jac_ODE_model,self.sim.y0,t)
                self.sim.run_simulation()
                ConvergeFlag = self.sim.SOLVER_CONVERGENCE
                                
            except ValueError:
                if not ConvergeFlag:
                    self.CONVERGENCE_FAILURE = True
                    self._reward = -100.0 #Discourage convergence failures using large penalty
                else:
                    self.RUNTIME_ERROR = True  
                    self._reward = 0.0 #Not a convergence failure
            
            assert ConvergeFlag,'Convergence flag should be true to calculate reward!'
                    
            self._reward = self.reward_calc()
            self.sim.tStart = self.sim.tStop
            self.update_reward_stats()
            
            if self.sim.tStop >= self.max_sim_time or self.CONVERGENCE_FAILURE or self.RUNTIME_ERROR:
                self.done = True
                
                if self.sim.tStop >= self.max_sim_time:
                    self.logger.info('{}:Simulation time limit exceeded in {} at tStop = {:.2f} s after completing {} steps - ending simulation!'.format(self.name,self.sim.name,self.sim.tStop, self._steps))
                elif self.CONVERGENCE_FAILURE:
                    self.logger.warning("{}:Convergence failure in {} at {:.2f} - ending simulation!!!".format(self.name,self.sim.name,self.sim.tStop))
                elif self.RUNTIME_ERROR:
                    self.logger.warning("{}:Run time error (due to error in code) in {} at {:.2f} - ending simulation!!!".format(self.name,self.sim.name,self.sim.tStop)) 
                    
            self._step_time = time.time() - time_start
            self.update_time_stats()
        
        return np.array(self.state), self._reward, self.done, {}

    def action_calc(self,action):
        """Calculate action"""
        
        assert action in self.action_space,'The action "{}" is not available in the environment action space!'.format(action)
        
        if not self.goals_list:   #Do nothing
            print('No goals were given to agent, so no control action is being taken and action is reset to 0.')
            action = 2
        else:
            goal_type = self.goals_list[0] #Only the first goal in goals list will be considered for now
        
        self.update_action_stats(action)
        
        if action is not 0: #Only proceed if action is not a do nothing action
            delQref = 0
            delVdcref = 0
            if action == 1:
                delQref = self._delQref #0.1e3                
            elif action == 2:
                delQref = -self._delQref #0.1e3                         
            elif action == 3:
                delVdcref = self._delVdcref #0.1 #Volts (DC)
            elif action == 4:
                delVdcref = -self._delVdcref #0.1 #Volts (DC)      

            #for action in self.env_goal_spec[goal_type]['action']['my_spec']:
            #    if action == 'Q_control':
            self.sim.PV_model.Q_ref = self.sim.PV_model.Q_ref + delQref/self.sim.PV_model.Sbase           

            #    if action == 'Vdc_control':
                    #elif 'power_regulation' in self.goals_list or 'Vdc_control' in self.goals_list:
            self.sim.PV_model.Vdc_ref = self.sim.PV_model.Vdc_ref + delVdcref/self.sim.PV_model.Vdcbase            
    
    def reward_calc(self):
        """Calculate reward"""
        
        goal_type = self.goals_list[0] #Only the first goal in goals list will be considered for now
            
        #if 'voltage_regulation' in self.goals_list:
        if goal_type == 'voltage_regulation':
            Qtarget = self.sim.PV_model.Q_ref
        #elif 'Q_control' in self.goals_list and 'voltage_regulation' not in self.goals_list:
        elif goal_type == 'Q_regulation':
            Qtarget = self.env_reward_spec['reference_values']['Q_ref']/self.sim.Sbase
        elif goal_type == 'power_regulation':        
        #if 'power_regulation' in self.goals_list or 'Vdc_control' in self.goals_list:
            Vdctarget = self.sim.PV_model.Vdc_ref
            Ptarget = self.env_reward_spec['reference_values']['P_ref']/self.sim.Sbase
        
        rewards = []
        
        for reward_type in self.env_goal_spec[goal_type]['reward']['my_spec']:
            
            if reward_type == 'Vdc_error':
            #if 'Vdc_control' in self.goals_list:
                if self.DISCRETE_REWARD:

                    if abs(self.sim.PV_model.Vdc - Vdctarget)/Vdctarget <= 0.02:
                        rewards.append(1)
                    elif abs(self.sim.PV_model.Vdc - Vdctarget)/Vdctarget >= 0.05:
                        rewards.append(-5)
                    else:
                        rewards.append(-1)
                else:
                    rewards.append(-(self.sim.PV_model.Vdc_ref -self.sim.PV_model.Vdc)**2)
            elif reward_type == 'Q_error':
            #if 'Q_control' in self.goals_list:
                if self.DISCRETE_REWARD:
                    if Qtarget == 0.0:
                        Qtarget = 1e-6
                    if abs(self.sim.PV_model.S_PCC.imag - Qtarget)/abs(Qtarget) <= 0.01:
                        rewards.append(1)
                    elif abs(self.sim.PV_model.S_PCC.imag - Qtarget)/abs(Qtarget) >= 0.05:
                        rewards.append(-5)
                    else:
                        rewards.append(-1)
                else:
                    rewards.append(-(self.sim.PV_model.S_PCC.imag - Qtarget)**2)
            elif reward_type == 'voltage_error':
            #if 'voltage_regulation' in self.goals_list:
                if self.DISCRETE_REWARD:

                    if abs(self.sim.PV_model.Vrms - self.sim.PV_model.Vrms_ref)/abs(self.sim.PV_model.Vrms_ref) <= 0.01:
                        rewards.append(1)
                    elif abs(self.sim.PV_model.Vrms - self.sim.PV_model.Vrms_ref)/abs(self.sim.PV_model.Vrms_ref) >= 0.05:
                        rewards.append(-5)
                    else:
                        rewards.append(-1)
                else:
                    rewards.append(-(self.sim.PV_model.Vrms - self.sim.PV_model.Vrms_ref)**2)                    

            #if 'power_regulation' in self.goals_list:
            elif reward_type == 'power_error':
                if self.DISCRETE_REWARD:
                    if abs(self.sim.PV_model.S_PCC.real - Ptarget)/Ptarget <= 0.01:
                        rewards.append(1)
                    elif abs(self.sim.PV_model.S_PCC.real - Ptarget)/Ptarget >= 0.03:
                        rewards.append(-5)
                    else:
                        rewards.append(-1)       
                else:
                    rewards.append(-(self.sim.PV_model.S_PCC.real - Ptarget)**2)

        return sum(rewards)
    
    def initialize_environment_variables(self):
        """Initialize environment variables."""
        
        self._steps  = 0
        self._reward = 0
        
        self.done = False
        self.CONVERGENCE_FAILURE = False
        self.RUNTIME_ERROR = False
        
        self.update_env_goal(goal_type=None,goal_spec=None)
        self.initialize_stats()
    
    def reset(self):
        """Reset environment."""
        
        self.logger.debug('{}:Resetting environment and creating new PV-DER simulation'.format(self.name))
        self.logger.debug('{}:Max environment steps:{}'.format(self.name,self.spec.max_episode_steps))
        
        #if self.sim in locals(): #Check if simulation object exists
        if hasattr(self, 'sim'):
            self.logger.debug('{}:The existing simulation object {} will be removed from environment.'.format(self.name,self.sim.name))
            self.cleanup_PVDER_simulation()
            
        else:
            self.logger.debug('{}:No simulation object exists in environment!'.format(self.name))
        
        self.initialize_environment_variables()
        self.setup_PVDER_simulation()

        self.logger.info('{}:Environment was reset and {} attached'.format(self.name,self.sim.name))
        return np.array(self.state)
    
    def render(self, mode='vector'):
        """Show observations from environment."""
        
        _observations = {'ma':np.array(self.sim.PV_model.ma),
                         'ia':np.array(self.sim.PV_model.ia*self.sim.Ibase),                  
                         'Vdc':np.array(self.sim.PV_model.Vdc*self.sim.PV_model.Vdcbase),
                         'vta':np.array(self.sim.PV_model.vta*self.sim.Vbase),
                         'va':np.array(self.sim.PV_model.va*self.sim.Vbase),
                         'Ppv':np.array(self.sim.PV_model.Ppv*self.sim.Sbase),
                         'S_inverter':np.array(self.sim.PV_model.S*self.sim.Sbase),
                         'S_PCC':np.array(self.sim.PV_model.S_PCC*self.sim.Sbase),
                         'Q_ref':np.array(self.sim.PV_model.Q_ref*self.sim.Sbase),
                         'Vdc_ref':np.array(self.sim.PV_model.Vdc_ref*self.sim.PV_model.Vdcbase),
                         'tStart':np.array(self.sim.tStart)
                        }
        
        for item in _observations:
                print('{}:{:.2f},'.format(item,_observations[item]), end=" ")
        print('\nReward:{:.5f}'.format(self._reward))
        
        self.show_step_time()
       
        if mode == 'human':
            self.results.plot_DER_simulation(plot_type='active_power_Ppv_Pac_PCC')
            self.results.plot_DER_simulation(plot_type='reactive_power_Q_PCC')
            self.results.plot_DER_simulation(plot_type='voltage_Vdc')
            self.results.plot_DER_simulation(plot_type='duty_cycle')
            self.results.plot_DER_simulation(plot_type='voltage_LV')
            self.results.plot_DER_simulation(plot_type='current')
        
    def setup_PVDER_simulation(self,model_type='model_2'):
        """Setup simulation environment."""
        
        self.max_sim_time = self.max_sim_time_user
        
        events = SimulationEvents(events_spec = self.env_events_spec,verbosity ='INFO')
        grid_model = Grid(events=events)
        
        PVDER_model = DERModel(modelType=self.env_model_spec[model_type]['DERModelType'],
                           events=events,configFile=self.env_model_spec[model_type]['configFile'],
                           gridModel=grid_model,
                           derId=self.env_model_spec[model_type]['derId'],
                           standAlone = True,steadyStateInitialization=True)   

        PVDER_model.DER_model.LVRT_ENABLE = False  #Disconnects PV-DER using ride through settings during voltage anomaly
        PVDER_model.DER_model.DO_EXTRA_CALCULATIONS = True
        #PV_model.Vdc_EXTERNAL = True
        self.sim = DynamicSimulation(gridModel=grid_model,PV_model=PVDER_model.DER_model,
                                     events = events,verbosity = 'INFO',solverType='odeint',
                                     LOOP_MODE=False) #'odeint','ode-vode-bdf'
        self.sim.jacFlag = True      #Provide analytical Jacobian to ODE solver
        self.sim.DEBUG_SOLVER = False #Give information on solver convergence
        
        self.results = SimulationResults(simulation = self.sim)
        self.results.PER_UNIT = False
        self.results.font_size = 18
        
        self.generate_simulation_events()
        PVDER_model.Qref_EXTERNAL = True  #Enable VAR reference manipulation through outside program
        self.sim.DEBUG_SOLVER = False #Give information on solver convergence
        self.sim.tInc = self.env_sim_spec['sim_time_step'] #1/60.0 #self.time_taken_per_step #     
        #self.sim.reset_stored_trajectories()
        #print(self.sim.t)        
   
    def generate_simulation_events(self):
        """Create random events."""
        
        events_type_list =[]
        for event_type in self.env_events_spec:
            if self.env_events_spec[event_type]['ENABLE']:
                events_type_list.append(event_type)
                
        self.sim.simulation_events.create_random_events(self.env_events_spec['voltage']['t_events_start'],
                                                        self.env_events_spec['voltage']['t_events_stop'],
                                                        self.env_events_spec['voltage']['t_events_step'],
                                                        events_type=events_type_list)    
    
    def update_env_events(self,event_spec_list):
        """Update environment simulation event."""
        
        assert isinstance(event_spec_list,list), 'event_spec_list should be a list!'
        print(isinstance(event_spec_list,list))
        
        for event_spec in event_spec_list:
            assert isinstance(event_spec,dict), 'Event spec should be a dictionary!'
            assert len(event_spec.keys()) == 1, 'Only one event type should be specified at at time!'
            
            event_type = list(event_spec)[0] 
            event_parameter_type_list = event_spec[event_type].keys()           
        
            if event_type in self.env_events_spec.keys():
                for event_parameter_type in event_parameter_type_list:
                    if event_parameter_type in self.env_events_spec[event_type].keys():
                        self.env_events_spec[event_type][event_parameter_type] = event_spec[event_type][event_parameter_type]
                    else:
                        raise ValueError('{} is not a valid paramter for {} event!'.format(event_parameter_type,event_type))
            else:
                raise ValueError('{} is not a valid event!'.format(event_type))
        
        if hasattr(self, 'sim'):
            self.logger.warning('{}:Updating events spec in attached simulation object {}.'.format(self.name,self.sim.name))
            self.sim.simulation_events.update_events_spec(events_spec = self.env_events_spec)
            self.generate_simulation_events()
    
    def cleanup_PVDER_simulation(self):
        """Remove previous instances."""
        
        del self.sim
    
    def update_env_goal(self,goal_type,goal_spec):
        """Update goal spec."""
        
        if goal_type is None and goal_spec is None:
            self.logger.debug('{}:Updating env goal spec with default values.'.format(self.name))
            for goal_type in self.env_goal_spec:
                self.env_goal_spec[goal_type]['reward']['my_spec'] = self.env_goal_spec[goal_type]['reward']['required']  
                self.env_goal_spec[goal_type]['action']['my_spec'] = self.env_goal_spec[goal_type]['action']['required']
            
        else:
            assert goal_spec.keys() in self.env_goal_spec.keys(), '{} is not a valid goal!'.format(goal_spec.keys())
            print('Update function is under construction!')
    
    def calc_returns(self):
        """Calculate returns."""
        
        _goals_list= list(self.env_goal_spec.keys())
        _action_spec_list =['random','inc','dec','no_change']
        _n_episodes = 2
        
        self.env_average_return = {}        
       
        for goal in _goals_list:
            self.env_average_return[goal] = {}
            self.goals_list = [goal]
            self.show_env_config()
            for action_spec in _action_spec_list:
                self.env_average_return[goal][action_spec] = {}
                total_return = 0.0
                print('Calculating average return for "{}" goal with "{}" action.'.format(goal,action_spec)) 
                for i in range(_n_episodes):
                    episode_return = 0.0
                    observation = self.reset()
                    done = False
                    while not done:
                        if action_spec == 'random':
                            action = self.action_space.sample()  #Sample actions from the environment's discrete action space
                        elif action_spec == 'inc':
                            action = 0
                        elif action_spec == 'dec':
                            action = 1
                        elif action_spec == 'no_change':
                            action = 2
                    
                        observation, reward, done, _ = self.step(action)
                        episode_return += reward
                    
                    total_return += episode_return
                self.env_average_return[goal][action_spec]['return'] =  total_return/_n_episodes
                self.env_average_return[goal][action_spec]['ref'] =  [self.sim.PV_model.Vdc_ref*self.sim.Vbase,
                                                                      self.sim.PV_model.Q_ref*self.sim.Sbase]
                
        self.show_env_returns()
    
    def show_env_config(self):
        """Show environment configuration."""
        
        print('Environment name:{}'.format(self.name))
        self.show_goal_config()
        self.show_sim_config()
    
    def show_sim_config(self):
        """Show actions."""
        
        print('time step:{:.4f},del Qref:{:.2f},del Vdcref:{:.2f}'.format(self._sim_time_per_env_step,self._delQref,self._delVdcref))
        print('Voltage events:{},Insolation events:{}'.format(self.env_events_spec['voltage']['ENABLE'],self.env_events_spec['insolation']['ENABLE']))
    
    def show_goal_config(self):
        """Show goal configuration."""
        
        goal_type = self.goals_list[0]
        print('Goal type:',goal_type)
        print('Reward type:{},Discrete:{}'.format(self.env_goal_spec[goal_type]['reward']['my_spec'],self.DISCRETE_REWARD))
        
        if goal_type == 'Q_regulation':
            print('Q reference:{}'.format(self.env_reward_spec['reference_values']['Q_ref']))
        elif goal_type == 'power_regulation':
            print('P reference:{}'.format(self.env_reward_spec['reference_values']['P_ref']))
                
        print('Action type:',self.env_goal_spec[goal_type]['action']['my_spec'])

    def show_env_returns(self):
        """Show environment returns."""
        
        self.pp.pprint(self.env_average_return)             
    
    @property                         #Decorator used for auto updating
    def state(self):
        """Create array containing states."""
        
        _state = (self.sim.PV_model.ia.real,self.sim.PV_model.ia.imag,
                  self.sim.PV_model.va.real,self.sim.PV_model.va.imag,
                  self.sim.PV_model.S_PCC.real,self.sim.PV_model.S_PCC.imag,
                  self.sim.PV_model.Vdc,self.sim.PV_model.Ppv,
                  self.sim.PV_model.Vdc_ref,self.sim.PV_model.Q_ref,
                  self.sim.tStart/self.max_sim_time)        
    
        return _state
    
    @property
    def max_sim_time(self):
        return self.__max_sim_time
    
    @property
    def goals_list(self):
        return self.__goals_list
    
    @property
    def DISCRETE_REWARD(self):
        return self.__DISCRETE_REWARD
    
    @property
    def n_sim_time_steps_per_env_step(self):
        return self.__n_sim_time_steps_per_env_step
    
    ## the attribute name and the method name must be same which is used to set the value for the attribute
    @max_sim_time.setter
    def max_sim_time(self,max_sim_time):
        
        if max_sim_time is None:
            self.__max_sim_time = self.spec.max_episode_steps*self._sim_time_per_env_step
        elif isinstance(max_sim_time, (int, float)):
            if max_sim_time < self.env_sim_spec['min_sim_time']:
                self.__max_sim_time = self.env_sim_spec['min_sim_time']
            elif max_sim_time > self.spec.max_episode_steps*self._sim_time_per_env_step: #self._max_episode_steps
                self.logger.warning('User specifed maximum simulation time is {} s, but allowable simulation time is only {} s!'.format(max_sim_time,self.spec.max_episode_steps*self._sim_time_per_env_step))
                self.__max_sim_time = self.spec.max_episode_steps*self._sim_time_per_env_step #self.max_episode_steps*
            else:
                self.__max_sim_time = max_sim_time
        else:
            raise ValueError("max_sim_time must be a float!")
            
    @goals_list.setter
    def goals_list(self,goals_list):
        
        if goals_list is None:
            self.__goals_list =  self.default_goals
        elif set(goals_list).issubset(self.env_goal_spec.keys()):
            self.__goals_list = goals_list
            self.logger.info('The goals list is updated, the goals are:{}'.format(self.goals_list))
        else:
             raise ValueError('Goal list:{} contains invalid elements, available elements are:{}'.format(goals_list,self.env_goal_spec.keys()))
                    
        return self.__goals_list    
    
    @DISCRETE_REWARD.setter
    def DISCRETE_REWARD(self,DISCRETE_REWARD):
        
        if DISCRETE_REWARD is None:
            self.__DISCRETE_REWARD =  self.env_reward_spec['DISCRETE_REWARD'] 
        elif isinstance(DISCRETE_REWARD, bool):
            self.__DISCRETE_REWARD = DISCRETE_REWARD
        else:
            raise ValueError("DISCRETE_REWARD must be a boolean!")
                    
        return self.__DISCRETE_REWARD

    @n_sim_time_steps_per_env_step.setter
    def n_sim_time_steps_per_env_step(self,n_sim_time_steps_per_env_step):
        
        if n_sim_time_steps_per_env_step is None:
            self.__n_sim_time_steps_per_env_step =  self.env_sim_spec['n_sim_time_steps_per_env_step']['default'] 
        elif isinstance(n_sim_time_steps_per_env_step, int):
            if n_sim_time_steps_per_env_step >= self.env_sim_spec['n_sim_time_steps_per_env_step']['min']:
                self.__n_sim_time_steps_per_env_step = n_sim_time_steps_per_env_step
            else:
                self.__n_sim_time_steps_per_env_step = self.env_sim_spec['n_sim_time_steps_per_env_step']['min']
                self.logger.warning('n_sim_time_steps_per_env_step should be greater than {}!'.format(self.env_sim_spec['n_sim_time_steps_per_env_step']['min']))
        else:
            raise ValueError("n_sim_time_steps_per_env_step must be an integer!")
        
        self._sim_time_per_env_step = self.env_sim_spec['sim_time_step']*self.n_sim_time_steps_per_env_step
        self._delQref = self.env_action_spec['delQref']*self.n_sim_time_steps_per_env_step
        self._delVdcref = self.env_action_spec['delVdcref']*self.n_sim_time_steps_per_env_step
        
        return self.__n_sim_time_steps_per_env_step