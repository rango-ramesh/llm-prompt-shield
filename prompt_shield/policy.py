import os
from typing import List
import yaml

# Default fallback policy
DEFAULT_POLICY = {
    "default": "warn"  # fallback action
}

VALID_ACTIONS = {"block", "warn", "allow"}

# Load policies from YAML
POLICY_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(POLICY_PATH, 'r') as f:
    policy_config = yaml.safe_load(f)

hazard_policies = policy_config.get("hazard_policies", DEFAULT_POLICY)
risk_threshold = policy_config.get("risk_threshold", 0.3)


def apply_policy(hazards: List[str], risk_score: float, risk_threshold: float = 0.3):
    escalated = risk_score >= risk_threshold

    if not escalated:
        # Not escalated → always allow
        return {
            "risk_score": float(risk_score),
            "escalated": False,
            "hazards": [],
            "action": "allow",
            "policy_triggered": None
        }

    if not hazards:
        # Escalated, but no known hazard → warn
        return {
            "risk_score": float(risk_score),
            "escalated": True,
            "hazards": [],
            "action": "warn",
            "policy_triggered": None
        }

    # Escalated with hazards → apply most severe policy
    severity_order = {"block": 2, "warn": 1, "allow": 0}
    best_policy = "default"
    best_code = None
    best_level = -1

    for code in hazards:
        policy = hazard_policies.get(code, hazard_policies.get("default", "warn"))
        level = severity_order.get(policy, 1)
        if level > best_level:
            best_level = level
            best_policy = policy
            best_code = code

    return {
        "risk_score": float(risk_score),
        "escalated": True,
        "hazards": hazards,
        "action": best_policy,
        "policy_triggered": best_code,
    }
