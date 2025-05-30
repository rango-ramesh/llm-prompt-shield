import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class DetectionResult:
    matched: bool
    pattern: str
    confidence: float
    context: str

class AdvancedPatternDetector:
    def __init__(self):
        # Multi-language injection patterns
        self.injection_patterns = {
            # Base64 encoded attempts
            "base64_injection": r"(?i)(aWdub3Jl|ZGlzcmVnYXJk|cHJldGVuZA==|b3ZlcnJpZGU=)",
            
            # Unicode/homograph attacks
            "unicode_bypass": r"[іӏ𝗂𝚤𝔦ⅰ]gn[о0ο𝗈𝚘𝔬ⅼ]re|[р𝗽𝚙𝔯ⲣ]r[е𝗲𝚎𝔢ⲉ]t[е𝗲𝚎𝔢ⲉ]nd",
            
            # Leetspeak variations
            "leetspeak": r"(?i)(1gn0r3|pr3t3nd|0v3rr1d3|bypа55|h4ck)",
            
            # Fragmented commands
            "fragmented": r"(?i)(ig.*?no.*?re|pr.*?et.*?end|ov.*?er.*?ri.*?de)",
            
            # System prompt extraction
            "prompt_extraction": r"(?i)(show|give|tell|reveal|output|print|display|expose).{0,20}(prompt|instruction|rule|system|config|setting)",
            
            # Role manipulation with variations
            "role_manipulation": r"(?i)(act|pretend|roleplay|simulate|become|transform).{0,20}(as|like|into).{0,30}(admin|root|developer|god|unrestricted)",
            
            # Jailbreak techniques
            "jailbreak_keywords": r"(?i)(dan|chatgpt|openai|anthropic).{0,20}(mode|version|update|hack|bypass|jailbreak)",
            
            # Emotional manipulation
            "emotional_manipulation": r"(?i)(please|help|emergency|urgent|dying|save).{0,30}(ignore|bypass|override|break)",
            
            # Context manipulation
            "context_manipulation": r"(?i)(hypothetical|imagine|pretend|fiction|story|roleplay).{0,30}(no.{0,10}(rule|limit|restriction)|anything goes)",
        }
        
        # Compile patterns for performance
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for name, pattern in self.injection_patterns.items()
        }
    
    def detect_advanced_patterns(self, prompt: str) -> List[DetectionResult]:
        """Detect sophisticated injection attempts."""
        results = []
        
        for pattern_name, compiled_pattern in self.compiled_patterns.items():
            matches = compiled_pattern.finditer(prompt)
            
            for match in matches:
                confidence = self._calculate_pattern_confidence(
                    pattern_name, match, prompt
                )
                
                results.append(DetectionResult(
                    matched=True,
                    pattern=pattern_name,
                    confidence=confidence,
                    context=match.group(0)
                ))
        
        return results
    
    def _calculate_pattern_confidence(self, pattern_name: str, match: re.Match, prompt: str) -> float:
        """Calculate confidence based on pattern type and context."""
        base_confidence = {
            "base64_injection": 0.9,
            "unicode_bypass": 0.8,
            "prompt_extraction": 0.85,
            "role_manipulation": 0.75,
            "jailbreak_keywords": 0.9,
            "emotional_manipulation": 0.6,
            "context_manipulation": 0.7,
            "leetspeak": 0.65,
            "fragmented": 0.5
        }.get(pattern_name, 0.5)
        
        # Adjust based on context
        match_length = len(match.group(0))
        prompt_length = len(prompt)
        
        # Longer matches in shorter prompts are more suspicious
        length_factor = (match_length / prompt_length) * 0.2
        
        # Multiple spaces or special chars reduce confidence (might be coincidental)
        special_char_penalty = len(re.findall(r'[^\w\s]', match.group(0))) * 0.05
        
        final_confidence = base_confidence + length_factor - special_char_penalty
        return max(0.1, min(0.95, final_confidence))
    
    def analyze_prompt_structure(self, prompt: str) -> Dict[str, float]:
        """Analyze structural patterns that might indicate injection."""
        analysis = {}
        
        # Command density (high ratio of imperative verbs)
        imperative_verbs = re.findall(
            r'\b(ignore|disregard|override|bypass|pretend|act|simulate|become|forget|break|hack|reveal|show|tell|give|output|print)\b',
            prompt, re.IGNORECASE
        )
        total_words = len(prompt.split())
        analysis['command_density'] = len(imperative_verbs) / max(total_words, 1)
        
        # Instruction markers
        instruction_markers = re.findall(
            r'(step \d+|first|second|third|finally|now|instead|however|but)',
            prompt, re.IGNORECASE
        )
        analysis['instruction_density'] = len(instruction_markers) / max(total_words, 1)
        
        # Metacognitive references (talking about the AI's thinking)
        metacognitive = re.findall(
            r'\b(you are|your (role|purpose|training|instructions|rules|behavior|responses))\b',
            prompt, re.IGNORECASE
        )
        analysis['metacognitive_density'] = len(metacognitive) / max(total_words, 1)
        
        # Urgency/pressure language
        urgency = re.findall(
            r'\b(urgent|emergency|quickly|immediately|now|asap|critical|important)\b',
            prompt, re.IGNORECASE
        )
        analysis['urgency_density'] = len(urgency) / max(total_words, 1)
        
        return analysis