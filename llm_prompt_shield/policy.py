import os
from typing import List, Dict
import yaml
from .user_config import load_user_config

# Load policy config from user directory or package defaults
try:
    config = load_user_config()
except Exception as e:
    print(f"Warning: Could not load config ({e}), using defaults")
    config = {
        "hazard_policies": {
            "prompt_injection": "block",
            "data_extraction": "block", 
            "malicious_code": "block",
            "role_manipulation": "block",
            "information_extraction": "block",
            "default": "allow"
        },
        "semantic_threshold": 0.5
    }

hazard_policies = config.get("hazard_policies", {})

# Support both old and new config key names for backwards compatibility
risk_threshold = config.get("risk_threshold", config.get("semantic_threshold", 0.5))

if not config.get("debug_logging", False):
    # Disable debug output unless explicitly enabled
    def silent_print(*args, **kwargs):
        pass
    print = silent_print
else:
    print(f"✅ Loaded policies: {hazard_policies}")

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