import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import sys
import os
import logging
import random

import importlib.util
import sys

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
from pvder import utility_functions


class PVDER(gym.Env):
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
    events_spec = {'insolation':{'delay':0.5,'time_step':1.0,'min':85.0,'max':100.0,'ENABLE':False},
                   'voltage':{'delay':1.0,'time_step':1.0,'min':0.95,'max':1.02,'ENABLE':True}}  #Time delay between events
    
    simulation_spec = {'simulation_time_step':(1/60),'simulation_time_per_environment_step': (1/60)*30}
    goal_list=['Q_control','voltage_regulation'] #List with agent goals. #'Vdc_deviation',
    
    #time_taken_per_step  = (1/60)*15 #Simulation time for each step in environment
    
    def __init__(self,max_time = 10.0,DISCRETE_REWARD=True):        
        """
        max_time: Scalar specifiying maximum simulation time in seconds for PV-DER model.
        goal_list: List with agent goals.
        """
        self.max_time = max_time
        #self.goal_list = goal_list
        self.DISCRETE_REWARD = DISCRETE_REWARD
        
        self.setup_PVDER_simulation()
        
        self.steps  = 0
        self.reward = 0
        
        self.done = False
        self.convergence_failure = False
        logging.getLogger().setLevel(logging.INFO)

    
    def step(self, action):
        
        if self.steps == 0:
            print('New episode started with tStart = {} s'.format(self.sim.tStart))
        
        if self.done == True and not self.convergence_failure:
            print('Simulation completed - Reset environment to start new simulation!')
        
        elif self.done == True and self.convergence_failure:
            print('Simulation stopped due to converge failure at {} s - Reset environment to start new simulation!'.format(self.sim.tStart))
        
        else:
            self.sim.tStop = self.sim.tStart + self.simulation_spec['simulation_time_per_environment_step'] #
            self.steps = self.steps +1
            #t = [self.sim.tStart,self.sim.tStop]
            t = self.sim.t_calc()
            
            if action == 0:
                    _Qref = - 0.1e3  #VAR
                    _Vdcref = -0.1 #Volts (DC)
            elif action == 1:
                    _Qref =  0.1e3
                    _Vdcref = 0.1 #Volts (DC)
            elif action == 2:
                    _Qref = 0.0
                    _Vdcref = 0.0 #Volts (DC)
            else:
                print('{} is an invalid action'.format(action))
            if 'Q_control' or 'voltage_regulation' in self.goal_list:
                self.sim.PV_model.Q_ref = self.sim.PV_model.Q_ref + _Qref/self.sim.PV_model.Sbase
                
            elif 'Ppv_control' in self.goal_list:
                self.sim.PV_model.Vdc_ref = self.sim.PV_model.Vdc_ref + _Vdcref/self.sim.PV_model.Vdcbase
            else:   #Do nothing
                print('no_control')
                pass
            
            try:
                solution,info = self.sim.call_ODE_solver(self.sim.ODE_model,self.sim.jac_ODE_model,self.sim.y0,t)
                
                #self.sim.run_simulation()
                
            except ValueError:
                self.convergence_failure = True
                self.reward = -100.0 #Discourage convergence failures using large penalty
            
            else:  #If solution converges calculate reward
                if self.sim.COLLECT_SOLUTION:
                    self.sim.collect_solution(solution,t)
                    
                self.reward = self.reward_calc()
                self.sim.tStart = self.sim.tStop                
            
            if self.sim.tStart >= self.max_time or self.convergence_failure:
                self.done = True
                if not self.convergence_failure:
                    print('Simulation time limit exceeded in {} at {:.2f} - ending simulation!'.format(self.sim.name,self.sim.tStart))
                else:
                    print("Convergence failure in {} at {:.2f} - ending simulation!!!".format(self.sim.name,self.sim.tStart))
        
        return np.array(self.state), self.reward, self.done, {}

    def reward_calc(self):
        """Calculate reward"""
        _Qtarget = self.sim.PV_model.Q_ref
        _Ppvtarget = 0.95
        _reward = []
        
        if 'Vdc_deviation' in self.goal_list:
            if abs(self.sim.PV_model.Vdc-self.sim.PV_model.Vdc_ref)/self.sim.PV_model.Vdc_ref <= 0.02:
                _reward.append(1)
            elif abs(self.sim.PV_model.Vdc-self.sim.PV_model.Vdc_ref)/self.sim.PV_model.Vdc_ref >= 0.05:
                _reward.append(-5)
            else:
                _reward.append(-1)
                                #self.reward = (self.sim.PV_model.Vdc_ref -self.sim.PV_model.Vdc)**2
                #self.reward = -(self.sim.PV_model.Ppv-0.85)**2 -(self.sim.PV_model.Vdc_ref -self.sim.PV_model.Vdc)**2
        
        if 'Q_control' in self.goal_list:
            if self.DISCRETE_REWARD:
                print(self.sim.PV_model.S_PCC.imag, _Qtarget)
                if abs(self.sim.PV_model.S_PCC.imag - _Qtarget)/abs(_Qtarget) <= 0.01:
                    _reward.append(1)
                elif abs(self.sim.PV_model.S_PCC.imag - _Qtarget)/abs(_Qtarget) >= 0.05:
                    _reward.append(-5)
                    
                else:
                    _reward.append(-1)
            else:
                _reward.append(-(self.sim.PV_model.S_PCC.imag - _Qtarget)**2)
        
        if 'voltage_regulation' in self.goal_list:
            if self.DISCRETE_REWARD:
                print(self.sim.PV_model.Vrms*500,self.sim.PV_model.Vrms_ref*500)
                if abs(self.sim.PV_model.Vrms - self.sim.PV_model.Vrms_ref)/abs(self.sim.PV_model.Vrms_ref) <= 0.01:
                    _reward.append(1)
                elif abs(self.sim.PV_model.Vrms - self.sim.PV_model.Vrms_ref)/abs(self.sim.PV_model.Vrms_ref) >= 0.05:
                    _reward.append(-5)
                    
                else:
                    _reward.append(-1)
            else:
                _reward.append(-(self.sim.PV_model.S_PCC.imag - _Qtarget)**2)
        
        if 'Ppv_control' in self.goal_list:
            if abs(self.sim.PV_model.Ppv - _Ppvtarget)/_Ppvtarget <= 0.01:
                _reward.append(1)
            if abs(self.sim.PV_model.Ppv - _Ppvtarget)/_Ppvtarget >= 0.03:
                _reward.append(-5)
            else:
                _reward.append(-1)       
        
        return sum(_reward)
    
    def reset(self):
        #self.state = 0
        print('----Resetting environment and creating new PV-DER simulation----')
        self.cleanup_PVDER_simulation()
        self.setup_PVDER_simulation()
        
        self.reward = 0.0
        self.done = False
        self.convergence_failure = False
        self.steps  = 0
        print('Environment was reset and {} attached'.format(self.sim.name))
        return np.array(self.state)
    
    def render(self, mode='vector'):
        
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
        
        events = SimulationEvents()
        grid = Grid(events=events)
        PV_model = SolarPV_DER_ThreePhase(grid_model = grid,events=events,
                                          standAlone = True, Sinverter_rated = 50.0e3,
                                          STEADY_STATE_INITIALIZATION=True) 
                               #gridVoltagePhaseA=(.50+0j),gridVoltagePhaseB=(-.25-.43301270j),gridVoltagePhaseC=(-.25+.43301270j),

        PV_model.LVRT_ENABLE = False  #Disconnects PV-DER using ride through settings during voltage anomaly
        PV_model.DO_EXTRA_CALCULATIONS = True
        #PV_model.Vdc_EXTERNAL = True
        self.sim = DynamicSimulation(PV_model=PV_model,events = events,grid_model=grid,LOOP_MODE = True,COLLECT_SOLUTION=True)
        self.results = SimulationResults(simulation = self.sim)
        self.results.PER_UNIT = False
        self.results.font_size = 18

        self.sim.jacFlag = True      #Provide analytical Jacobian to ODE solver
        self.sim.DEBUG_SOLVER = False #Give information on solver convergence
        
        #self.sim.tInc = 1/120.0
        """
        self.sim.tStop = 4.0
        self.sim.LOOP_MODE = False
        self.sim.run_simulation()  #Run simulation to bring environment to steady state
        self.sim.LOOP_MODE = True
        """
        self.create_random_events()
        PV_model.Qref_EXTERNAL = True  #Enable VAR reference manipulation through outside program
        self.sim.DEBUG_SOLVER = False #Give information on solver convergence
        self.sim.tInc = self.simulation_spec['simulation_time_step'] #1/60.0 #self.time_taken_per_step #     
        self.sim.reset_stored_trajectories()
        #print(self.sim.t)
   
    def create_random_events(self):
        """Create random events."""
        
        if 'insolation' in self.events_spec.keys() and self.events_spec['insolation']['ENABLE']:
            t_events = np.arange(self.events_spec['insolation']['delay'],self.max_time,self.events_spec['insolation']['time_step'])
            for t in t_events:
                insolation = self.events_spec['insolation']['min'] + random.random()*(self.events_spec['insolation']['max']-self.events_spec['insolation']['min'])
                self.sim.simulation_events.add_solar_event(t,insolation)
        
        if 'voltage' in self.events_spec.keys() and self.events_spec['voltage']['ENABLE']:
            t_events = np.arange(self.events_spec['voltage']['delay'],self.max_time,self.events_spec['voltage']['time_step'])
            for t in t_events:
                voltage = self.events_spec['voltage']['min'] + random.random()*(self.events_spec['voltage']['max']-self.events_spec['voltage']['min']) 
                self.sim.simulation_events.add_grid_event(t,voltage)
    
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
                  self.sim.tStart/self.max_time)        
    
        return _state