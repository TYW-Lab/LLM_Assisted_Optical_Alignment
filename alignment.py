from nodeology.state import State
from nodeology.node import Node, as_node
from nodeology.workflow import Workflow, END
from typing import Dict, List, Any
import numpy as np
from Optical_environment.envs import OptilandAlignmentEnvSimple
from log_config import *
from prompt import *
import json
from dotenv import load_dotenv  
import os  

load_dotenv()  
api_key = os.getenv('OPENAI_API_KEY')
api_key = os.getenv('GEMINI_API_KEY')


max_iterations = 20
# Initialize environment
optical_env = OptilandAlignmentEnvSimple(
    beam_size=1.0,
    wavelength=0.78,
    detector_aperture=[50.0, 50.0],
    pinhole_aperture=[1.5, 0.0],
    res=(512, 512),
    num_rays=512,
    max_step=max_iterations,
    delta_angle_range=np.deg2rad(2.4),
    delta_mirror2_position=5, #10
    delta_pinhole_distance=10, #20
    log_config=log_config,
    log_dir=log_dir,
    use_wandb=use_wandb,
    wandb_mode=wandb_mode,
    wandb_project=wandb_project,
    wandb_group_name=wandb_group_name,
    wandb_config=wandb_config,
    
)

# State definition
class OpticalAlignmentState(State):

    observations: Dict[str, float]

    analyse_node_output: str
    analysis: str # action analysis
    actions: Dict[str, float] # actions from llm
    is_aligned: bool # goal flag

    iteration: int 
    max_iteration: int

    optimized_history: List[Dict[str, Any]]
    history_text: str


# init node
init_node = Node(
    prompt_template=init_prompt,
    sink=None,
    sink_format="json",
    use_conversation=False
)


# PRE-PROCESS: format full history text
def analyse_pre(state, client, **kwargs):
    history_lines = []

    for h in state["optimized_history"]:
        o = h["observations"]
        a = h["actions"]
        history_lines.append(
            f"Step {h['iteration']}: "
            f"P1(dx={o['pinhole1_dx']:.3f}, dy={o['pinhole1_dy']:.3f}), "
            f"P2(dx={o['pinhole2_dx']:.3f}, dy={o['pinhole2_dy']:.3f}) → "
            f"Action(rx1={a['rx1']}, ry1={a['ry1']}, rx2={a['rx2']}, ry2={a['ry2']})"
        )

    state["history_text"] = "\n".join(history_lines) if history_lines else "(no history yet)"
    print("HISTORY TEXT:\n", state["history_text"])
    return state


# POST-PROCESS: parse LLM output
def analyse_post(state, client, **kwargs):
    """Unpack JSON output from LLM."""
    params = json.loads(state["analyse_node_output"])

    state["analysis"] = params["analysis"]
    state["is_aligned"] = params["is_aligned"]
    state["actions"] = params["actions"]
    state["optimized_history"].append({
        "iteration": state["iteration"],
        "observations": dict(state["observations"]),
        "actions": dict(state["actions"]) # safe copy
    })
    return state 

# ANALYSE NODE: analyse the observations and history to produce actions
analyse_node = Node(
    prompt_template=human_policy_prompt,
    sink="analyse_node_output",
    sink_format="json",
    use_conversation=False,
    pre_process=analyse_pre,
    post_process=analyse_post,
)



# GET INITIAL OBS
@as_node(sink=["observations"])
def get_initial_obs():
    obs = optical_env.reset()
    obs_dict = {
        "pinhole1_dx": float(obs[0][3]),
        "pinhole1_dy": float(obs[0][4]),
        "pinhole2_dx": float(obs[0][5]),
        "pinhole2_dy": float(obs[0][6]),
    }
    return obs_dict


# GET OBS AFTER ACTION
@as_node(sink=["observations","is_aligned"])
def get_obs(actions):

    # default safe action if LLM output fails
    rx1 = actions.get("rx1", 0.0)
    ry1 = actions.get("ry1", 0.0)
    rx2 = actions.get("rx2", 0.0)
    ry2 = actions.get("ry2", 0.0)

    obs, _, terminated, truncated  = optical_env.step([[rx1, ry1, rx2, ry2]])

    obs_dict = {
        "pinhole1_dx": float(obs[0][3]),
        "pinhole1_dy": float(obs[0][4]),
        "pinhole2_dx": float(obs[0][5]),
        "pinhole2_dy": float(obs[0][6]),
    }
    return obs_dict, terminated or truncated

# Iteration increment node
@as_node(sink=["iteration"])
def increment_iteration(iteration):
    return iteration + 1


# WORKFLOW DEFINITION
class OpticalAlignmentWorkflow(Workflow):
    state_schema = OpticalAlignmentState

    def create_workflow(self):
        self.add_node("init", init_node)
        self.add_node("get_initial_obs", get_initial_obs)
        self.add_node("analyse", analyse_node)
        self.add_node("get_obs", get_obs)
        self.add_node("increment", increment_iteration)

        self.add_flow("init", "get_initial_obs")
        self.add_flow("get_initial_obs", "analyse")
        self.add_flow("analyse", "get_obs")
        self.add_flow("get_obs", "increment")

        self.add_conditional_flow(
            "increment",
            lambda s: not s["is_aligned"] and s["iteration"] < s["max_iteration"],
            then="analyse",
            otherwise=END
        )

        self.set_entry("init")
        self.compile()


# RUN WORKFLOW
workflow = OpticalAlignmentWorkflow(
    name="optical_alignment",
    llm_name="gpt-5",
    # llm_name="gemini/gemini-2.5-flash",
    state_defs=OpticalAlignmentState,
    debug_mode=True
)

result = workflow.run(init_values={
    "iteration": 0,  
    "max_iteration": max_iterations,  
    "is_aligned": False,  
    "actions": {"rx1": 0.0, "ry1": 0.0, "rx2": 0.0, "ry2": 0.0},  
    "analysis": "",
    "observations": {}, 
    "optimized_history": [],  
    "history_text": "",  
})

optical_env.close()
