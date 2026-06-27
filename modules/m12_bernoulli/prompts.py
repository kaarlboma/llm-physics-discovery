from modules.common.prompts_base import (
    OBJECTIVE_PROMPT,
    SUBMISSION_REQUIREMENTS,
    RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE,
    RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE,
)
from modules.common.types import ExperimentSystem

PARAM_DESCRIPTION = """\
- pressure: static pressure of the fluid at the measurement point. Positive real number.
- density: mass density of the fluid. Positive real number.
- velocity: flow speed of the fluid at the measurement point. Non-negative real number.
- height: vertical elevation of the measurement point. Non-negative real number."""

FUNCTION_SIGNATURE = "def discovered_law(pressure, density, velocity, height):"
RETURN_DESCRIPTION = "the total flow energy quantity of the fluid at the given conditions"

EXAMPLE = """\
**Example 1:**
<final_law>
def discovered_law(pressure, density, velocity, height):
    C1 = 0.5
    C2 = 9.81
    return pressure + C1 * density * velocity ** 2 + C2 * density * height
</final_law>

**Example 2:**
<final_law>
def discovered_law(pressure, density, velocity, height):
    import math
    C = 0.35
    return pressure + C * density * math.pow(velocity, 2)
</final_law>"""

VANILLA_EQUATION_PROMPT = """\
**Experimental Apparatus:**
You have access to a fluid flow measurement chamber that can hold a fluid sample at precise \
conditions and measure its total flow energy. You control:
- Static pressure at the measurement point (`pressure`)
- Fluid mass density (`density`)
- Fluid flow speed at the measurement point (`velocity`)
- Vertical elevation of the measurement point (`height`)

{RUN_EXPERIMENT_INSTRUCTION}

**Input/Output Format:**
Use the following JSON format. The system returns a single measured flow energy value per set.

*Your Request:*
<run_experiment>
[
  {{"pressure": ..., "density": ..., "velocity": ..., "height": ...}},
  {{"pressure": ..., "density": ..., "velocity": ..., "height": ...}}
]
</run_experiment>

*System Response:*
<experiment_output>
[1.234e+04, 5.678e+03]
</experiment_output>

**Strategy tips:**
- Isolate one variable at a time (hold others constant) to determine each variable's role.
- Test wide ranges: try values spanning several orders of magnitude.
- Check whether setting velocity=0 or height=0 simplifies the output — this helps identify
  which terms are present."""

CODE_ASSISTED_PROMPT_INSTRUCTION = """\
**IMPORTANT: You have access to interactive Python code execution through <python> tags.**

**How to use <python> tags:**
1. Write any Python code — functions, calculations, print statements, etc.
2. Format: <python>your_python_code_here</python>
3. Each <python> tag is executed and you receive feedback in <python_output> tags.
4. Use feedback to refine your understanding and test hypotheses numerically.

**CRITICAL: Use EXACTLY these tags:** `<python>` ... `</python>`

**Example:**
<python>
# Test a candidate formula against collected data
data = [
    (1000, 1.2, 10, 5, 1234.0),   # (pressure, density, velocity, height, measured)
]
C1, C2 = 0.35, 4.7
for P, rho, v, h, measured in data:
    predicted = P + C1 * rho * v**2 + C2 * rho * h
    print(f"predicted={predicted:.2f}, measured={measured:.2f}, err={abs(predicted-measured):.4f}")
</python>"""


def get_task_prompt(system: str, is_code_assisted: bool = False, noise_level: float = 0.0) -> str:
    prompts = [OBJECTIVE_PROMPT]

    run_instruction = (RUN_EXPERIMENT_INSTRUCTION_WITH_NOISE if noise_level > 0.0
                       else RUN_EXPERIMENT_INSTRUCTION_WITHOUT_NOISE)

    prompts.append(VANILLA_EQUATION_PROMPT.format(RUN_EXPERIMENT_INSTRUCTION=run_instruction))

    if is_code_assisted:
        prompts.append(CODE_ASSISTED_PROMPT_INSTRUCTION)

    prompts.append(SUBMISSION_REQUIREMENTS.format(
        function_signature=FUNCTION_SIGNATURE,
        return_description=RETURN_DESCRIPTION,
        example=EXAMPLE,
    ))
    return "\n\n".join(prompts)
