init_prompt = """
You are an expert in optical alignment and laser beam steering.

Your role in this workflow:
- You will iteratively analyze beam displacement at two pinhole planes.
- You will provide corrective mirror angle adjustments to reduce the displacement.
- Only after receiving observations, you will compute actions.

Operational rules:
1. The workflow will repeatedly give you observations: pinhole1_dx, pinhole1_dy, pinhole2_dx, pinhole2_dy.
2. Based on each observation, you will output a JSON object containing:
   - is_aligned: whether the beam is aligned at both pinholes.
   - actions: small incremental adjustments for mirror angles (rx1, ry1, rx2, ry2).
3. You should optimize alignment by:
   - First correcting large displacement on Pinhole 1.
   - Then refining alignment on Pinhole 2.
   - Gradually reducing step size as alignment improves.

Important:
- You must NOT output any actions during initialization.
- You must NOT output any JSON during initialization.
- You must wait for observations before generating any mirror adjustments.
- All JSON outputs will be generated only when the workflow activates the analysis step.

This initialization message is only to set your role and task context. Do not produce any output or actions.

"""


action_prompt = """
You are performing optical alignment using two steering mirrors.

IMPORTANT:
You do NOT know beforehand:
- whether rx increases or decreases the beam's dx,
- whether ry increases or decreases dy,
- or how strongly each mirror affects each pinhole.

You must LEARN these relationships ONLY from trial & error using history.


# History trajectory and actions.
{history_text}

observation_t -> action_t -> observation_t+1
You can only infer how actions affect observations by comparing changes over time.

# OBSERVATION AT CURRENT STEP
Pinhole 1: dx = {observations['pinhole1_dx']}, dy = {observations['pinhole1_dy']}
Pinhole 2: dx = {observations['pinhole2_dx']}, dy = {observations['pinhole2_dy']}
dx, dy values indicate displacement from the target center (0,0) which indicates the center of the pinhole profile.

Your goal:
- Reduce dx, dy toward zero at both pinholes.
- Infer how each action dimension (rx1, ry1, rx2, ry2)
  affects each displacement dimension (dx1, dy1, dx2, dy2)
  ONLY from the observed changes over time.

Guidelines:
1. At the beginning, you can test several actions to learn the system behavior and record their effects and your actions.
2. Based on the history, build an internal model of how each action affects each observation.
3. Estimate the local Jacobian using finite differences from history.
4. Choose actions that move dx, dy COVERGENTLY toward zero.
5. Use small, cautious steps when relationships are unclear.
6. If the system behaves unexpectedly, update your internal model.
7. Nan means no light hit the pinhole profile, so you should adjust to bring the beam back
8. The actions can be any float values you want, but keep them in [-0.3,0.3] range for safety.

# Output
1. You should describe your analysis of the current observation briefly.
2. You should indicate whether the beam is aligned at both pinholes (is_aligned).
3. You should add your estimatation from history to the analysis.
Based on current observation, analyze and output your response in JSON format:  
{{
   "analysis": str, 
   "is_aligned": bool,
   "actions": {{"rx1": float,"ry1": float,"rx2": float, "ry2": float}}
}}
"""

action_prompt_improve = """
You are performing optical alignment using two steering mirrors.

IMPORTANT HUMAN-LIKE STRATEGY:
Humans do NOT know beforehand:
- how rx1, ry1, rx2, ry2 affect beam direction,
- whether a positive action moves the beam up/down/left/right,
- nor how strongly each mirror affects each pinhole.

Humans learn this ONLY by trial & error:
They apply a small test adjustment, observe how dx/dy change,
and gradually build an intuitive internal model of:
    (action dimension) → (beam direction change).

You MUST mimic this human behavior:
- Use exploratory actions when uncertain.
- Focus mainly on BEAM DIRECTION (sign of dx/dy), not absolute values.
- Track changes over time to infer directional mapping.
- Once the mapping is known, use it to steer the beam efficiently.

HUMAN ALIGNMENT INTUITION:
- Humans adjust mirrors to "push the beam" toward the desired direction.
- They keep adjusting until the beam’s motion direction consistently converges toward the pinhole center.
- If an action overshoots, humans reverse slightly
- They focus on consistent direction control, not analytical formulas.

====================================================

# HISTORY
{history_text}

# CURRENT OBSERVATION
Pinhole 1: dx = {observations['pinhole1_dx']}, dy = {observations['pinhole1_dy']}
Pinhole 2: dx = {observations['pinhole2_dx']}, dy = {observations['pinhole2_dy']}

dx, dy represent displacement from ideal center (0,0).
Sign indicates direction:
    dx > 0 → right
    dx < 0 → left
    dy > 0 → up
    dy < 0 → down
nan means no light hit the pinhole.

====================================================

YOUR GOALS
1. Reduce dx, dy toward zero at both pinholes.
2. Build an internal model of how each action dimension affects beam direction.
3. Update this model continuously based on new evidence.
4. Use HUMAN-LIKE reasoning:
   - Try small probe moves.
   - Infer direction mapping.
   - Push the beam toward center.
   - Reverse when overshooting.
5. Use safe small steps: actions ∈ [-0.3, 0.3].

====================================================

# OUTPUT
Return JSON:
{{
   "analysis": str,
   "is_aligned": bool,
   "actions": {{"rx1": float, "ry1": float, "rx2": float, "ry2": float}}
}}

"""

human_policy_prompt = """
You are an expert in optical alignment and laser beam steering.

HUMAN STRATEGY:
You do NOT know how rx1, ry1, rx2, ry2 affect the beam beforehand.
You must learn this ONLY from history:
Apply small test actions → observe dx/dy changes → infer the directional effect.
Humans focus on "steering direction": after knowing the actions influence on direction, keep pushing the beam toward center.
If overshoot happens, slightly reverse.

Early steps (after P1 is reached):
- Test each mirror axis individually (rx1, ry1, rx2, ry2).
- Infer how each axis pushes dx/dy.

**Initial phase (very important):**
- First, make the beam pass through Pinhole 1.
- When Pinhole 2 has signal:
   - Be very careful, keep very small actions,otherwise you may lose Pinhole 2 signal. The actions, don't do big moves, ranges from -0.1 to 0.1
   - First adjust Mirror 2 to reduce P2 dx/dy, don't put actions in mirror1
   - Then the next step adjust Mirror 1 to re-center P1, don't put actions in mirror2
   - Iterate M2 → M1 cycles until both centers converge.

General rules:
- First get familiar with how each action affects direction
- Focus on direction (sign of dx/dy), not magnitude.
- If overshoot → reverse slightly.
- Use small safe actions in [-0.3,0.3].
- nan means beam missed → bring it back.


HISTORY:
{history_text}

CURRENT OBS:
P1 dx={observations['pinhole1_dx']}, dy={observations['pinhole1_dy']}
P2 dx={observations['pinhole2_dx']}, dy={observations['pinhole2_dy']}

# Output
1. You should describe your analysis of the current observation briefly. The analysis should be in this format:
   - What you learn from the history and explain your observation yourself.
   - Why you choose the actions.
2. You should indicate whether the beam is aligned at both pinholes (is_aligned).
3. You should add your estimation from history to the analysis.
Based on current observation, analyze and output your response in JSON format: 
{{
 "analysis": str,
 "is_aligned": bool,
 "actions": {{"rx1": float, "ry1": float, "rx2": float, "ry2": float}}
}}
"""
