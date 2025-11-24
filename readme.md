# Optical Alignment Environment

## Overview
The project aims to let LLMs to do the optical alignment. The optical system is a Two-Mirror-Two-Pinhole-Alignment-System

## Installation

### Prerequisites 
> [Installation instrustion provided by Optiland](https://optiland.readthedocs.io/en/latest/installation.html)
> [Nodeology](https://github.com/xyin-anl/Nodeology)
```bash
pip install -r requirements.txt
```
## Introduction

### Project directory
```
LLM_Assisted_Optical_Alignment              
├─ Optical_environment                      
│  ├─ envs                                  
│  │  ├─ Optical_alignment_env.py           
│  │  ├─ Optical_alignment_env_simple.py    
│  │  └─ __init__.py                        
│  ├─ Optical_System                        
│  │  ├─ RL_Optical_Sys.py                  
│  │  └─ __init__.py                        
│  ├─ tools                                       
│  │  ├─ visualize.py                       
│  │  └─ __init__.py                        
│  ├─ utils                                 
│  │  ├─ computing.py                       
│  │  ├─ helper.py                          
│  │  ├─ logger.py                          
│  │  ├─ visualize.py                       
│  │  └─ __init__.py                        
│  ├─ wrappers                              
│  │  ├─ rl_simple_env_wrapper.py           
│  │  └─ __init__.py                        
│  └─ __init__.py                           
├─ alignment.py                                                         
├─ log_config.py                            
├─ prompt.py                                
├─ readme.md                                
├─ sys_check.ipynb                          
└─ visualize.py                                                      
```
You need to create an `.env` file to import your API key.

### Environment
> The environments are in `/Optical_environment/` directory. The basic environment is `/Optical_environment/Optical_alignment_env.py`

#### Observation

For the simple environment, the observation is (1, 7)
>`[m2dx, m2dy, hit_m2, p1dx, p1dy, p2dx, p2dy]`

You can define any kind of observation based on the basic environment.

#### Reset
The optical system has 4 different configuration("Z" shape path and "U" shape path). So in order to cover all the configuration and make the agent more generalizable, we random pick a configuration in the reset.

And the Reset workflow is:
1. Pick one specific configuration.
2. Add bias angle to original mirror angles and add bias distance to original optical path.
3. return to original mirror angles.
4. `Step 0` -> give a random action.

#### Step

Get an actions in shape (1, 4) and get observations. The actions is ranging from -1.0 to +1.0 which are corresponding to -4 to +4 degree.

#### Logger

You can set log config to control the log. If you don't want to log every step or every episode, just change the number of `log_every_n_episodes=1, log_every_n_steps=1`. If you want to change the log parameters, you can go into the code and change the log in `step` and `episode`, and then change the log configuration.

### Visualize

See the `visualize.py` to render.

### System Check

Run the `sys_check.ipynb` to check the packages.

### Main
Run `alignment.py` to let the LLM do the alignment.