import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import random

import sys
import os
import logging
import importlib.util

# For illustrative purposes.
package_name = 'pvder'

spec = importlib.util.find_spec(package_name)
if spec is None:
    print(package_name +" is not installed! Please install the module from https://github.com/sibyjackgrove/SolarPV-DER-simulation-utility")

#np.set_printoptions(precision=2)  #for setting number of decimal places when printing numpy arrays

from pvder.DER_components_single_phase  import SolarPV_DER_SinglePhase  #Python 3 need full path to find module
from pvder.DER_components_three_phase  import SolarPV_DER_ThreePhase
from pvder.grid_components import Grid
from pvder.simulation_events import SimulationEvents
from pvder.simulation_utilities import SimulationResults
from pvder.dynamic_simulation import DynamicSimulation
from pvder.utility_classes import Logging

from pvder import utility_functions


class PVDER(gym.Env,Logging):
    metadata = {'render.modes': ['vector','human']}
    
    observed_quantities = ['iaR','iaI',
                           'vaR','vaI',
                           'P_PCC','Q_PCC',
                           'Vdc','Ppv',
                           'Vdc_ref','Q_ref',
                           'tStart'] #,'maR','maI'
    action_space = spaces.Discrete(3)
    observation_space = spaces.Box(low=-10, high=10, shape=(len(observed_quantities),),dtype=np.float32)
    
    #objective_dict = {'Q_ref_+':{'Q_ref':0.1},'Q_ref_-':{'Q_ref':-0.1}}
    env_model_spec = {'SinglePhase':True}
    
    env_events_spec = {'insolation':{'t_events_start':1.0,'t_events_stop':25.0,'t_events_step':1.0,'min':85.0,'max':100.0,'ENABLE':False},
                       'voltage':{'t_events_start':1.0,'t_events_stop':25.0,'t_events_step':1.0,'min':0.98,'max':1.02,'ENABLE':True}} 
    
    env_sim_spec = {'sim_time_step':(1/60),'sim_time_per_env_step': (1/60)*30,'min_sim_time':1.0}    
    
    env_reward_spec={'goals_list':['voltage_regulation'],
                     'valid_goals_list':['voltage_regulation','power_regulation','Q_control','Vdc_control'],
                     'reference_values':{'P_ref':0.95},
                     'DISCRETE_REWARD':True
                    } #List with agent goals. 
    
    def __init__(self, n_sim_time_steps_per_env_step = 30, max_sim_time=None,
                 DISCRETE_REWARD=None, goals_list=None,
                 verbosity='INFO'):   
        """
        Setup PV-DER environment variables.
        
        Args:
           n_sim_time_steps_per_env_step (int): Scalar specifiying simulation time steps per environment step.           
           max_sim_time (float): Scalar specifiying maximum simulation time in seconds for simulation within the environment.
           goals_list (list): List of strings specifying agent goals.
           verbosity (string): Specify logging levels - ('DEBUG','INFO').
        """
        
        self.name = 'DER_env_1'
        
        self.env_sim_spec['sim_time_per_env_step'] = self.env_sim_spec['sim_time_step']*n_sim_time_steps_per_env_step
        self.max_sim_time_user = max_sim_time
        
        self.goals_list = goals_list
        self.DISCRETE_REWARD = DISCRETE_REWARD
        
        self.initialize_environment_variables()
        
        self.initialize_logger()
        self.verbosity = verbosity #Set logging level - {DEBUG,INFO,WARNING,ERROR}
    
    def __del__(self):
        # body of destructor
        print("Destructor called - environment deleted!")
    
    def step(self, action):
        
        if self.steps == 0:
            print('New episode started with tStart = {} s'.format(self.sim.tStart))
        
        if self.done and self.sim.tStop >= self.max_sim_time:
            print('Simulation completed - Reset environment to start new simulation!')
        
        elif self.done and self.CONVERGENCE_FAILURE:
            print('Simulation stopped due to converge failure at {} s - Reset environment to start new simulation!'.format(self.sim.tStop))
        
        elif self.done and self.RUNTIME_ERROR:
            print('Simulation stopped due to run time error failure at {} s - Check code for error in logic!'.format(self.sim.tStop))
        
        elif not self.done:
            self.sim.tStop = self.sim.tStart + self.env_sim_spec['sim_time_per_env_step'] #
            self.steps = self.steps +1
            #t = [self.sim.tStart,self.sim.tStop]
            t = self.sim.t_calc()

            self.action_calc(action)
            try:
                solution,info,ConvergeFlag = self.sim.call_ODE_solver(self.sim.ODE_model,self.sim.jac_ODE_model,self.sim.y0,t)
                
                #self.sim.run_simulation()
                
            except ValueError:
                if not ConvergeFlag:
                    self.CONVERGENCE_FAILURE = True
                    self.reward = -100.0 #Discourage convergence failures using large penalty
                else:
                    self.RUNTIME_ERROR = True  
                    self.reward = 0.0 #Not a convergence failure
            
            else:  #If solution converges calculate reward
                assert ConvergeFlag,'Convergence flag should be true to calculate reward!'
                
                if self.sim.COLLECT_SOLUTION:
                    self.sim.collect_solution(solution,t)
                    
                self.reward = self.reward_calc()
                self.sim.tStart = self.sim.tStop                
            
            if self.sim.tStop >= self.max_sim_time or self.CONVERGENCE_FAILURE or self.RUNTIME_ERROR:
                self.done = True
                
                if self.sim.tStop >= self.max_sim_time:
                    print('Simulation time limit exceeded in {} at tStop = {:.2f} s after completing {} steps - ending simulation!'.format(self.sim.name,self.sim.tStop, self.steps))
                elif self.CONVERGENCE_FAILURE:
                    print("Convergence failure in {} at {:.2f} - ending simulation!!!".format(self.sim.name,self.sim.tStop))
                elif self.RUNTIME_ERROR:
                    print("Run time error (due to error in code) in {} at {:.2f} - ending simulation!!!".format(self.sim.name,self.sim.tStop))             
        
        return np.array(self.state), self.reward, self.done, {}

    def action_calc(self,action):
        """Calculate action"""
        
        assert action in self.action_space,'The action "{}" is not available in the environment action space!'.format(action)
        
        if action == 0:
            _Qref = - 0.1e3  #VAR
            _Vdcref = -0.1 #Volts (DC)
        elif action == 1:
            _Qref =  0.1e3
            _Vdcref = 0.1 #Volts (DC)
        elif action == 2:
            _Qref = 0.0
            _Vdcref = 0.0 #Volts (DC)
            
        if 'voltage_regulation' or 'Q_control' in self.goals_list:
            self.sim.PV_model.Q_ref = self.sim.PV_model.Q_ref + _Qref/self.sim.PV_model.Sbase
                
        elif 'power_regulation' or 'Vdc_control' in self.goals_list:
            self.sim.PV_model.Vdc_ref = self.sim.PV_model.Vdc_ref + _Vdcref/self.sim.PV_model.Vdcbase
        
        else:   #Do nothing
            print('No goals were given to agent, so no control action is being taken.')
    
    def reward_calc(self):
        """Calculate reward"""
        
        _Qtarget = self.sim.PV_model.Q_ref
        _Vdctarget = self.sim.PV_model.Vdc_ref
        _Ptarget = self.env_reward_spec['reference_values']['P_ref']
        _reward = []
        
        if 'Vdc_control' in self.goals_list:
            if self.DISCRETE_REWARD:
                
                if abs(self.sim.PV_model.Vdc - _Vdctarget)/_Vdctarget <= 0.02:
                    _reward.append(1)
                elif abs(self.sim.PV_model.Vdc - _Vdctarget)/_Vdctarget >= 0.05:
                    _reward.append(-5)
                else:
                    _reward.append(-1)
            else:
                _reward.append(-(self.sim.PV_model.Vdc_ref -self.sim.PV_model.Vdc)**2)
                
        if 'Q_control' in self.goals_list:
            if self.DISCRETE_REWARD:
                
                if abs(self.sim.PV_model.S_PCC.imag - _Qtarget)/abs(_Qtarget) <= 0.01:
                    _reward.append(1)
                elif abs(self.sim.PV_model.S_PCC.imag - _Qtarget)/abs(_Qtarget) >= 0.05:
                    _reward.append(-5)
                else:
                    _reward.append(-1)
            else:
                _reward.append(-(self.sim.PV_model.S_PCC.imag - _Qtarget)**2)
        
        if 'voltage_regulation' in self.goals_list:
            if self.DISCRETE_REWARD:
                
                if abs(self.sim.PV_model.Vrms - self.sim.PV_model.Vrms_ref)/abs(self.sim.PV_model.Vrms_ref) <= 0.01:
                    _reward.append(1)
                elif abs(self.sim.PV_model.Vrms - self.sim.PV_model.Vrms_ref)/abs(self.sim.PV_model.Vrms_ref) >= 0.05:
                    _reward.append(-5)
                else:
                    _reward.append(-1)
            else:
                _reward.append(-(self.sim.PV_model.Vrms - self.sim.PV_model.Vrms_ref)**2)
        
        if 'power_regulation' in self.goals_list:
            if abs(self.sim.PV_model.Ppv - _Ppvtarget)/_Ppvtarget <= 0.01:
                _reward.append(1)
            if abs(self.sim.PV_model.Ppv - _Ppvtarget)/_Ppvtarget >= 0.03:
                _reward.append(-5)
            else:
                _reward.append(-1)       
        
        return sum(_reward)
    
    def initialize_environment_variables(self):
        """Initialize environment variables."""
        
        self.steps  = 0
        self.reward = 0
        
        self.done = False
        self.CONVERGENCE_FAILURE = False
        self.RUNTIME_ERROR = False
    
    def reset(self):
        """Reset environment."""
        
        self.logger.debug('----Resetting environment and creating new PV-DER simulation----')
        self.logger.debug('Max environment steps:',self.spec.max_episode_steps)
        
        #if self.sim in locals(): #Check if simulation object exists
        if hasattr(self, 'sim'):
            self.logger.debug('The existing simulation object {} will be removed from environment.'.format(self.sim.name))
            self.cleanup_PVDER_simulation()
            
        else:
            self.logger.debug('No simulation object exists in environment!')
        
        self.initialize_environment_variables()
        self.setup_PVDER_simulation()

        self.logger.info('Environment was reset and {} attached'.format(self.sim.name))
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
        print('\nReward:{:.4f}'.format(self.reward))
       
        if mode == 'human':
            self.results.plot_DER_simulation(plot_type='active_power_Ppv_Pac_PCC')
            self.results.plot_DER_simulation(plot_type='reactive_power_Q_PCC')
            self.results.plot_DER_simulation(plot_type='voltage_Vdc')
            self.results.plot_DER_simulation(plot_type='duty_cycle')
            self.results.plot_DER_simulation(plot_type='voltage_LV')
            self.results.plot_DER_simulation(plot_type='current')
        
    def setup_PVDER_simulation(self):
        """Setup simulation environment."""
        
        self.max_sim_time = self.max_sim_time_user
        
        events = SimulationEvents(events_spec = self.env_events_spec,verbosity ='INFO')
        grid_model = Grid(events=events)
        
        if self.env_model_spec['SinglePhase']:
            PVDER_model = SolarPV_DER_SinglePhase(grid_model = grid_model,events=events,
                                                 Sinverter_rated = 10.0e3,
                                                 standAlone = True,STEADY_STATE_INITIALIZATION=True,
                                                 verbosity ='INFO')
        else:
            PVDER_model = SolarPV_DER_ThreePhase(grid_model = grid_model,events=events,
                                                 Sinverter_rated = 50.0e3,
                                                 standAlone = True,STEADY_STATE_INITIALIZATION=True,
                                                 verbosity ='INFO')                                

        PVDER_model.LVRT_ENABLE = False  #Disconnects PV-DER using ride through settings during voltage anomaly
        PVDER_model.DO_EXTRA_CALCULATIONS = True
        #PV_model.Vdc_EXTERNAL = True
        self.sim = DynamicSimulation(PV_model=PVDER_model,events = events,grid_model=grid_model,
                                     LOOP_MODE = True,COLLECT_SOLUTION=True,
                                     verbosity ='INFO')
        self.sim.jacFlag = True      #Provide analytical Jacobian to ODE solver
        self.sim.DEBUG_SOLVER = False #Give information on solver convergence
        
        self.results = SimulationResults(simulation = self.sim)
        self.results.PER_UNIT = False
        self.results.font_size = 18
        
        self.generate_simulation_events()
        PVDER_model.Qref_EXTERNAL = True  #Enable VAR reference manipulation through outside program
        self.sim.DEBUG_SOLVER = False #Give information on solver convergence
        self.sim.tInc = self.env_sim_spec['sim_time_step'] #1/60.0 #self.time_taken_per_step #     
        self.sim.reset_stored_trajectories()
        #print(self.sim.t)
   
    def generate_simulation_events(self):
        """Create random events."""
        
        self.sim.simulation_events.create_random_events(self.env_events_spec['insolation']['t_events_start'],
                                                        self.env_events_spec['insolation']['t_events_stop'],
                                                        self.env_events_spec['insolation']['t_events_step'],
                                                        events_type=['insolation','voltage'])
    
    def cleanup_PVDER_simulation(self):
        """Remove previous instances."""
        
        del self.sim
    
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
    
    ## the attribute name and the method name must be same which is used to set the value for the attribute
    @max_sim_time.setter
    def max_sim_time(self,max_sim_time):
        
        if max_sim_time is None:
            self.__max_sim_time = self.spec.max_episode_steps*self.env_sim_spec['sim_time_per_env_step']
        elif isinstance(max_sim_time, (int, float)):
            if max_sim_time < self.env_sim_spec['min_sim_time']:
                self.__max_sim_time = self.env_sim_spec['min_sim_time']
            elif max_sim_time > self.spec.max_episode_steps*self.env_sim_spec['sim_time_per_env_step']: #self._max_episode_steps
                self.logger.warning('User specifed maximum simulation time is {} s, but allowable simulation time is only {} s!'.format(max_sim_time,self.spec.max_episode_steps*self.env_sim_spec['sim_time_per_env_step']))
                self.__max_sim_time = self.spec.max_episode_steps*self.env_sim_spec['sim_time_per_env_step'] #self.max_episode_steps*
            else:
                self.__max_sim_time = max_sim_time
        else:
            raise ValueError("max_sim_time must be a float!")
            
    @goals_list.setter
    def goals_list(self,goals_list):
        
        if goals_list is None:
            self.__goals_list =  self.env_reward_spec['goals_list'] 
        elif set(goals_list).issubset(self.env_reward_spec['valid_goals_list']):
            self.__goals_list = goals_list       
        else:
             raise ValueError('Goal list:{} contains invalid elements, available elements are:{}'.format(goals_list,self.reward_spec['goals_list']))
                    
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
