**Status:** Maintenance (expect bug fixes and minor updates)
# Gym environment for PV-DER

[![CodeFactor](https://www.codefactor.io/repository/github/sibyjackgrove/gym-solarpvder-environment/badge/master)](https://www.codefactor.io/repository/github/sibyjackgrove/gym-solarpvder-environment/overview/master)

Solar photovoltaic distributed energy resources (PV-DER) are power electronic inverter based generation (IBG) connected to the electric power distribution system (eg. roof top solar PV systems). This environment consists of a single DER connected to a stiff voltage source as shown in the following schematic:

![schematic of PV-DER](PVDER_schematic.png)

## Basics
The dynamics of the DER are modelled using dynamic phasors. One step in the environment is equivalent to **n** simulation time steps and each simulation time step is one half-cycle (1/120 s).
### Events in the environment
There are two types of perturbations in the environment: a) Change in grid voltage magnitude b) Change in solar insolation.

## Installation
First install the [Solar PVDER simulation utility.](https://github.com/sibyjackgrove/SolarPV-DER-simulation-utility) Then you can install the PVDER environment using following commands:
```
git clone https://github.com/sibyjackgrove/gym-SolarPVDER-environment.git
cd gym-SolarPVDER-environment
pip install -e .
```
Other dependencies: OpenAI Gym, Numpy, SciPy
## Using the environment
The environment can be instantiated just like any other OpenAI Gym environment as show below:
```
import gym
import gym_PVDER
env = gym.make('PVDER-v0')
```
### Available user options
* DISCRETE_REWARD: If this is set to *True* discrete rewards (i.e. +1, -1, -5 etc) will be returned every time step. Otherwise reward will be the error between goals and actual values.
 
### Try it out in Google Colab:
Create environment and make a random agent interact with it:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sibyjackgrove/gym-SolarPVDER-environment/blob/master/examples/gym_PVDER_environment_import_test.ipynb)

The environment can also be used with TensorFlow agents. Try out the demo below which trains a DQN agent using the environment:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sibyjackgrove/gym-SolarPVDER-environment/blob/master/examples/gym_PVDER_environment_tf_agents_DQN_demo.ipynb)
## Citation
If you use this code please cite it as:
```
@misc{gym-PVDER,
  title = {{gym-SolarPVDER-environment}: A environment for solar photovoltaic distributed energy resources},
  author = "{Siby Jose Plathottam}",
  howpublished = {\url{https://github.com/sibyjackgrove/gym-SolarPVDER-environment}},
  url = "https://github.com/sibyjackgrove/gym-SolarPVDER-environment",
  year = 2019,
  note = "[Online; accessed 18-March-2019]"
}
```
