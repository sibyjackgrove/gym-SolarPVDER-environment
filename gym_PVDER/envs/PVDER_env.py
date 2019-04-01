import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import sys
import os
import logging

#sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))+'\\pvder') #Otherwise Python 3 won't find module

np.set_printoptions(precision=2)  #for setting number of decimal places when printing numpy arrays

from gym_PVDER.envs.pvder.DER_components import SolarPV_DER  #Python 3 need full path to find module
from gym_PVDER.envs.pvder.grid_components import Grid
from gym_PVDER.envs.pvder.simulation_events import SimulationEvents
from gym_PVDER.envs.pvder.dynamic_simulation import GridSimulation
from gym_PVDER.envs.pvder import utility_functions

class PVDER(gym.Env):
    metadata = {'render.modes': ['human']}
    
    observed_quantities = ['iaR','iaI',
                           'vaR','vaI',
                           'P_PCC','Q_PCC',
                           'Vdc','Ppv',
                           'Vdc_ref','Q_ref',
                           'tStart'] #,'maR','maI'
    action_space = spaces.Discrete(3)
    observation_space = spaces.Box(low=-10, high=10, shape=(len(observed_quantities),),dtype=np.float32)
    
    objective_dict = {'Q_ref_+':{'Q_ref':0.1},'Q_ref_-':{'Q_ref':-0.1}}
    
    def __init__(self,max_time = 10.0,goal_list=['Vdc_deviation','Q_control'],DISCRETE_REWARD=True):        
        """
        max_time: Scalar specifiying maximum simulation time in seconds for PV-DER model.
        goal_list: List with agent goals.
        """
        self.max_time = max_time
        self.goal_list = goal_list
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
            self.sim.tStop = self.sim.tStart + self.sim.tInc
            self.steps = self.steps +1
            if action == 0:
                    _Qref = -0.5e3  #VAR
                    _Vdcref = -0.1 #Volts (DC)
            elif action == 1:
                    _Qref = 0.5e3
                    _Vdcref = 0.1 #Volts (DC)
            else:
                    _Qref = 0.0
                    _Vdcref = 0.0 #Volts (DC)
                    
            #self.sim.PV_model.Q_ref = self.sim.PV_model.Q_ref + _Qref/self.sim.PV_model.Sbase
            self.sim.PV_model.Vdc_ref = self.sim.PV_model.Vdc_ref + _Vdcref/self.sim.PV_model.Vdcbase
            try:
                self.sim.call_ODE_solver(self.sim.ODE_model,self.sim.jac_ODE_model,self.sim.y0,[self.sim.tStart,self.sim.tStop])
            except ValueError:
                self.convergence_failure = True
                self.reward = -100.0 #Discourage convergence failures using large penalty

            else:
                self.reward = self.reward_calc()

                                    
                self.sim.tStart = self.sim.tStop                
            
            if self.sim.tStart >= self.max_time or self.convergence_failure:
                self.done = True
                if not self.convergence_failure:
                    print('Simulation time limit exceeded in {} at {:.2f} - ending simulation!'.format(self.sim.name,self.sim.tStart))
                else:
                    print("Convergence failure in {} at {:.2f} - ending simulation!!!".format(self.sim.name,self.sim.tStart))
                    self.render()
        
        return np.array(self.state), self.reward, self.done, {}

    def reward_calc(self):
        """Calculate reward"""
        _Qtarget = 0.1
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
                if abs(self.sim.PV_model.S_PCC.imag - _Qtarget)/_Qtarget <= 0.01:
                    _reward.append(1)
                elif abs(self.sim.PV_model.S_PCC.imag - _Qtarget)/_Qtarget >= 0.05:
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
    
    def render(self, mode='human'):
        
        _state = {'ia':np.array(self.sim.PV_model.ia),
                  'ma':np.array(self.sim.PV_model.ma),
                  'Vdc':np.array(self.sim.PV_model.Vdc),
                  'Ppv':np.array(self.sim.PV_model.Ppv),
                  'S_inverter':np.array(self.sim.PV_model.S),
                  'S_PCC':np.array(self.sim.PV_model.S_PCC),
                  'Q_ref':np.array(self.sim.PV_model.Q_ref),
                  'Vdc_ref':np.array(self.sim.PV_model.Vdc_ref),
                  'tStart':np.array(self.sim.tStart)
                 }
        print('State:{},Reward:{:.4f}'.format(_state,self.reward))
        
    def setup_PVDER_simulation(self):
        """Setup simulation environment."""
        
        events = SimulationEvents()
        grid = Grid(events=events,standAlone=False)
        PV_model = SolarPV_DER(grid_model=grid,events=events,
                               standAlone=False,Sinverter_rated = 50.0e3,
                               gridVoltagePhaseA=(.50+0j),
                               gridVoltagePhaseB=(-.25-.43301270j),
                               gridVoltagePhaseC=(-.25+.43301270j),
                               STEADY_STATE_INITIALIZATION=False)

        PV_model.LVRT_ENABLE = False  #Disconnects PV-DER using ride through settings during voltage anomaly
        PV_model.VAR_EXTERNAL = False
        PV_model.Vdc_EXTERNAL = True
        self.sim = GridSimulation(grid_model=grid,PV_model=PV_model,simulation_events = events)
        self.sim.jacFlag = True      #Provide analytical Jacobian to ODE solver
        self.sim.DEBUG_SOLVER = False #Give information on solver convergence
        
        self.sim.tInc = 1/120.0
        
        self.sim.tStop = 4.0
        self.sim.run_simulation()  #Run simulation to bring environment to steady state
        self.sim.DEBUG_SOLVER = False #Give information on solver convergence
        self.sim.tInc = 1/60.0        
        #print(self.sim.t)
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