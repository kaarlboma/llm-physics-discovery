import re
import numpy as np
from typing import Union
from utils.noise import inject_noise
from modules.common.evaluation import evaluate_law as shared_evaluate_law
from .laws import get_ground_truth_law

ABSOLUTE_PRECISION = 1e-3


def validate_function_definition(code: str):
    if not re.search(r'def\s+discovered_law\s*\(pressure,\s*density,\s*velocity,\s*height\):', code):
        return False, "Invalid function signature"
    if not re.search(r'return\s+.+', code):
        return False, "No return statement found"
    return True, None


def run_experiment_for_module(
    pressure: float,
    density: float,
    velocity: float,
    height: float,
    noise_level: float,
    difficulty: str = 'easy',
    system: str = 'vanilla_equation',
    law_version: str = None,
    **kwargs,
) -> float:
    force_law, _ = get_ground_truth_law(difficulty, law_version)
    true_value = force_law(pressure, density, velocity, height)
    return inject_noise(true_value, noise_level, ABSOLUTE_PRECISION)


def evaluate_law(
    llm_function_str: str,
    param_description: str,
    difficulty: str = 'easy',
    law_version: str = None,
    judge_model_name: str = 'ch45',
    trial_info=None,
) -> dict:
    is_valid, err = validate_function_definition(llm_function_str)
    if not is_valid:
        return {
            'rmsle': float('nan'),
            'exact_accuracy': 0.0,
            'symbolic_equivalent': False,
            'symbolic_msg': err,
            'error': err,
        }

    gt_law, _ = get_ground_truth_law(difficulty, law_version)
    num_points = 5000

    test_data = {
        'pressure': np.exp(np.random.uniform(np.log(1e2), np.log(1e5), num_points)),
        'density':  np.exp(np.random.uniform(np.log(1),   np.log(1e3), num_points)),
        'velocity': np.exp(np.random.uniform(np.log(0.1), np.log(1e2), num_points)),
        'height':   np.exp(np.random.uniform(np.log(0.1), np.log(1e2), num_points)),
    }

    parameter_mapping = {
        'pressure': 'pressure',
        'density':  'density',
        'velocity': 'velocity',
        'height':   'height',
    }

    return shared_evaluate_law(
        llm_function_str=llm_function_str,
        gt_law=gt_law,
        test_data=test_data,
        parameter_mapping=parameter_mapping,
        param_description=param_description,
        judge_model_name=judge_model_name,
        trial_info=trial_info,
        symbolic_check=True,
    )
