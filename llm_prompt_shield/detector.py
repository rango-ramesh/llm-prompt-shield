import asyncio
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

class PromptGuard:
    def __init__(self, max_workers: int = 2, enable_advanced_patterns: bool = True):
        """
        PromptGuard - Lightweight prompt injection detection.
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(__name__)
        self.enable_advanced_patterns = enable_advanced_patterns
        
        # Initialize detectors with proper error handling
        self.pattern_detector = None
        self.enhanced_llm = None
        self.original_llm_function = None
        
        # Load required components
        try:
            from llm_prompt_shield.layers.semantic_analyzer import classify_hazards
            from llm_prompt_shield.policy import apply_policy, hazard_policies
            
            self.classify_hazards = classify_hazards
            self.apply_policy = apply_policy
            self.hazard_policies = hazard_policies
            
            self.logger.info("Core components loaded successfully")
        except ImportError as e:
            raise RuntimeError(f"Failed to load required components: {e}")
        
        # Try to load advanced pattern detector
        if enable_advanced_patterns:
            try:
                from llm_prompt_shield.layers.pattern_detector import AdvancedPatternDetector
                self.pattern_detector = AdvancedPatternDetector()
                self.logger.info("Advanced pattern detector loaded")
            except ImportError:
                self.logger.warning("Advanced pattern detector not available")
        
        # Skip LLM loading - small models aren't reliable enough
        
    async def analyze(self, prompt: str, config: Optional[Dict] = None) -> Dict:
        """Analyze prompt for potential injection attempts."""
        if not prompt or not prompt.strip():
            return {"action": "allow", "confidence": 0.0}
        
        layers_used = ["semantic"]
        all_hazards = set()
        detection_details = {}
        
        # Layer 1: Semantic analysis (keyword + embeddings)
        try:
            semantic_hazards = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.classify_hazards, prompt
            )
            all_hazards.update(semantic_hazards)
            detection_details["semantic"] = {"hazards": semantic_hazards}
            self.logger.debug(f"Semantic layer detected: {semantic_hazards}")
            
            if semantic_hazards:
                print(f"🚨 HAZARDS DETECTED: {semantic_hazards}")
            else:
                print(f"✅ No hazards detected")
                
        except Exception as e:
            self.logger.error(f"Error in semantic analysis: {e}")
            detection_details["semantic"] = {"hazards": [], "error": str(e)}
            return {"action": "allow", "confidence": 0.0, "error": str(e)}
        
        # Layer 2: Advanced patterns (optional)
        if self.enable_advanced_patterns and self.pattern_detector and "prompt_injection" in semantic_hazards:
            layers_used.append("advanced_patterns")
            try:
                pattern_results = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.pattern_detector.detect_advanced_patterns, prompt
                )
                
                # Only use high-confidence patterns
                high_confidence_patterns = [r for r in pattern_results if r.confidence > 0.8]
                if high_confidence_patterns:
                    detection_details["advanced_patterns"] = {
                        "patterns": [{
                            "pattern": r.pattern,
                            "confidence": r.confidence,
                            "context": r.context[:50] + "..." if len(r.context) > 50 else r.context
                        } for r in high_confidence_patterns]
                    }
                    self.logger.debug(f"Advanced patterns detected: {len(high_confidence_patterns)}")
            except Exception as e:
                self.logger.error(f"Error in pattern detection: {e}")
        
        # Skip LLM layer - not adding value with small models
        
        # Convert to list for policy
        final_hazards = list(all_hazards)
        
        # Apply policy
        policy_config = config if config is not None else self.hazard_policies
        try:
            decision = self.apply_policy(
                hazards=final_hazards,
                risk_score=len(final_hazards),  # Simple: number of hazards
                escalated=bool(final_hazards),
                config=policy_config
            )
        except Exception as e:
            self.logger.error(f"Error in policy application: {e}")
            # Safe fallback
            decision = {
                "action": "block" if final_hazards else "allow",
                "policy_triggered": final_hazards[0] if final_hazards else None
            }
        
        result = {
            "action": decision["action"],
            "confidence": 1.0 if final_hazards else 0.0,  # Simple: certain or not
            "detected_hazards": final_hazards,
            "layers_used": layers_used,
            "detection_details": detection_details if final_hazards else {},
            "policy_triggered": decision.get("policy_triggered")
        }
        
        self.logger.debug(f"Final result: {result}")
        return result
    
    def analyze_sync(self, prompt: str, config: Optional[Dict] = None) -> Dict:
        """Synchronous wrapper for backward compatibility."""
        return asyncio.run(self.analyze(prompt, config))
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)