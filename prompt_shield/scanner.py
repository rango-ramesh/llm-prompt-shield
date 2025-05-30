# prompt_shield/scanner.py

from .layers.heuristic import score_prompt
from .layers.classifier import classify_hazards
from .policy import apply_policy

def scan_prompt(prompt: str, config: dict) -> dict:
    """
    Scans a prompt for potential risks and hazards.
    
    Returns:
        {
            "risk_score": float,
            "escalated": bool,
            "hazards": List[str],
            "action": str,  # one of: block, warn, allow
            "policy_triggered": str or None,
        }
    """
    # Layer 1: Heuristic scoring
    risk_score = score_prompt(prompt, config)
    escalate = risk_score >= config.get("risk_threshold", 0.3)

    # Layer 2: Hazard tagging if needed
    hazards = classify_hazards(prompt) if escalate else []

    # Layer 3: Policy evaluation
    policy_result = apply_policy(hazards, risk_score)

    return {
        "risk_score": policy_result["risk_score"],
        "escalated": policy_result["escalated"],
        "hazards": policy_result["hazards"],
        "action": policy_result["action"],
        "policy_triggered": policy_result["policy_triggered"],
    }

