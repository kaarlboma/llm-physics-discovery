import numpy as np
import random
from typing import Callable, List, Optional, Tuple

# Hidden alien-universe constants (not real-world values)
_C1 = 0.347   # alien kinetic-energy coefficient (real Bernoulli uses 0.5)
_C2 = 4.731   # alien gravitational-like constant
_C3 = 2.156   # extra constant used in hard variants


# ── Easy ──────────────────────────────────────────────────────────────────────

def _easy_v0(pressure: float, density: float, velocity: float, height: float) -> float:
    """B = C1 * density * velocity^2 + pressure  (no height term)"""
    try:
        return _C1 * density * velocity ** 2 + pressure
    except Exception:
        return float('nan')


def _easy_v1(pressure: float, density: float, velocity: float, height: float) -> float:
    """B = C1 * density * velocity^1.5 + pressure  (fractional velocity exponent)"""
    try:
        if velocity < 0:
            return float('nan')
        return _C1 * density * velocity ** 1.5 + pressure
    except Exception:
        return float('nan')


def _easy_v2(pressure: float, density: float, velocity: float, height: float) -> float:
    """B = C1 * density^1.5 * velocity^2 + pressure  (fractional density exponent)"""
    try:
        if density < 0:
            return float('nan')
        return _C1 * density ** 1.5 * velocity ** 2 + pressure
    except Exception:
        return float('nan')


# ── Medium ────────────────────────────────────────────────────────────────────

def _medium_v0(pressure: float, density: float, velocity: float, height: float) -> float:
    """B = C1 * density * velocity^2 + pressure + C2 * density * height"""
    try:
        return _C1 * density * velocity ** 2 + pressure + _C2 * density * height
    except Exception:
        return float('nan')


def _medium_v1(pressure: float, density: float, velocity: float, height: float) -> float:
    """B = C1 * density * velocity^2.5 + pressure + C2 * density * height"""
    try:
        if velocity < 0:
            return float('nan')
        return _C1 * density * velocity ** 2.5 + pressure + _C2 * density * height
    except Exception:
        return float('nan')


def _medium_v2(pressure: float, density: float, velocity: float, height: float) -> float:
    """B = C1 * density^1.5 * velocity^2 + pressure + C2 * height^1.5"""
    try:
        if density < 0 or height < 0:
            return float('nan')
        return _C1 * density ** 1.5 * velocity ** 2 + pressure + _C2 * height ** 1.5
    except Exception:
        return float('nan')


# ── Hard ──────────────────────────────────────────────────────────────────────

def _hard_v0(pressure: float, density: float, velocity: float, height: float) -> float:
    """B = C1 * density * velocity^2 + pressure + C2 * (density * height)^1.5"""
    try:
        if density < 0 or height < 0:
            return float('nan')
        return _C1 * density * velocity ** 2 + pressure + _C2 * (density * height) ** 1.5
    except Exception:
        return float('nan')


def _hard_v1(pressure: float, density: float, velocity: float, height: float) -> float:
    """B = C1 * density * velocity^3 + pressure^1.5 + C2 * density * height^2"""
    try:
        if pressure < 0 or height < 0:
            return float('nan')
        return _C1 * density * velocity ** 3 + pressure ** 1.5 + _C2 * density * height ** 2
    except Exception:
        return float('nan')


def _hard_v2(pressure: float, density: float, velocity: float, height: float) -> float:
    """B = C1 * density^2 * velocity^2 + pressure + C2 * height + C3 * density * height^0.5"""
    try:
        if density < 0 or height < 0:
            return float('nan')
        return (_C1 * density ** 2 * velocity ** 2
                + pressure
                + _C2 * height
                + _C3 * density * height ** 0.5)
    except Exception:
        return float('nan')


# ── Registry ──────────────────────────────────────────────────────────────────

LAW_REGISTRY = {
    'easy':   {'v0': _easy_v0,   'v1': _easy_v1,   'v2': _easy_v2},
    'medium': {'v0': _medium_v0, 'v1': _medium_v1, 'v2': _medium_v2},
    'hard':   {'v0': _hard_v0,   'v1': _hard_v1,   'v2': _hard_v2},
}


def get_ground_truth_law(difficulty: str, law_version: Optional[str] = None) -> Tuple[Callable, str]:
    if difficulty not in LAW_REGISTRY:
        raise ValueError(f"Invalid difficulty: {difficulty}. Choose from {list(LAW_REGISTRY)}")
    available = list(LAW_REGISTRY[difficulty])
    if law_version is None:
        law_version = random.choice(available)
    elif law_version not in available:
        raise ValueError(f"Version '{law_version}' not found for '{difficulty}'. Available: {available}")
    return LAW_REGISTRY[difficulty][law_version], law_version


def get_available_law_versions(difficulty: str) -> List[str]:
    if difficulty not in LAW_REGISTRY:
        raise ValueError(f"Invalid difficulty: {difficulty}")
    return list(LAW_REGISTRY[difficulty])
