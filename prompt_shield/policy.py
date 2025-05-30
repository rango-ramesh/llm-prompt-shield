import os
from typing import List, Dict
import yaml

# Load policy config
POLICY_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# Handle missing config file gracefully

with open(POLICY_PATH, "r") as f:
    config = yaml.safe_load(f)


hazard_policies = config.get("hazard_policies", {})
risk_threshold = config.get("risk_threshold", 0.3)


def apply_policy(hazards: List[str], risk_score: float, escalated: bool, config: Dict) -> Dict:
    """Apply policy decisions based on detected hazards."""
    # Use the passed config parameter
    policies = config
    
    # Ensure we have a default policy
    if "default" not in policies:
        policies["default"] = "allow"

    for hazard in hazards:
        normalized = hazard.strip().lower()
        action = policies.get(normalized, policies.get("default", "allow"))

        if action == "block":
            return {
                "risk_score": risk_score,
                "escalated": escalated,
                "hazards": hazards,
                "action": "block",
                "policy_triggered": hazard
            }

    return {
        "risk_score": risk_score,
        "escalated": escalated,
        "hazards": hazards,
        "action": "allow",
        "policy_triggered": None
    }