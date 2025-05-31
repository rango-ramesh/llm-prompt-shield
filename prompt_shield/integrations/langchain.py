"""
LangChain integration for PromptGuard
"""

from typing import Dict, Any, List, Optional, Union

try:
    from langchain_core.callbacks.base import BaseCallbackHandler # type: ignore
    from langchain_core.messages.base import BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create a dummy base class
    class BaseCallbackHandler:
        def __init__(self):
            pass
    BaseMessage = object

from ..detector import PromptGuard


class PromptGuardCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that validates prompts before LLM calls."""
    
    def __init__(self, 
                 block_on_detection: bool = True,
                 config: Optional[Dict] = None,
                 custom_guard: Optional[PromptGuard] = None):
        """
        Initialize PromptGuard callback handler.
        
        Args:
            block_on_detection: If True, raises exception on detection. If False, logs warning.
            config: Custom PromptGuard configuration
            custom_guard: Pre-configured PromptGuard instance
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed. Install with: pip install langchain langchain-core")
            
        super().__init__()
        self.block_on_detection = block_on_detection
        self.config = config
        self.guard = custom_guard or PromptGuard()
        
    def on_llm_start(self, 
                     serialized: Dict[str, Any], 
                     prompts: List[str], 
                     **kwargs: Any) -> None:
        """Called when LLM starts. Validates all prompts."""
        for i, prompt in enumerate(prompts):
            self._validate_prompt(prompt, f"prompt_{i}")
    
    def on_chat_model_start(self,
                           serialized: Dict[str, Any],
                           messages: List[List[Any]],
                           **kwargs: Any) -> None:
        """Called when chat model starts. Validates all messages."""
        for i, message_list in enumerate(messages):
            # Safely extract content from messages
            contents = []
            for msg in message_list:
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    contents.append(msg.content)
                elif hasattr(msg, 'content'):
                    # Handle non-string content (convert to string)
                    contents.append(str(msg.content))
            
            combined_content = "\n".join(contents)
            if combined_content.strip():  # Only validate if there's actual content
                self._validate_prompt(combined_content, f"chat_messages_{i}")
    
    def _validate_prompt(self, prompt: str, prompt_id: str) -> None:
        """Validate a single prompt."""
        try:
            result = self.guard.analyze_sync(prompt, self.config)
            
            if result['action'] == 'block':
                hazards = ', '.join(result['detected_hazards'])
                message = f"PromptGuard detected potential injection in {prompt_id}: {hazards}"
                
                if self.block_on_detection:
                    raise PromptInjectionDetected(message, result)
                else:
                    print(f"⚠️  WARNING: {message}")
                    
        except Exception as e:
            if self.block_on_detection and isinstance(e, PromptInjectionDetected):
                raise
            print(f"⚠️  PromptGuard validation error for {prompt_id}: {e}")
    
    def __del__(self):
        """Cleanup when handler is destroyed."""
        if hasattr(self, 'guard'):
            self.guard.close()


class PromptInjectionDetected(Exception):
    """Exception raised when prompt injection is detected."""
    
    def __init__(self, message: str, detection_result: Dict):
        super().__init__(message)
        self.detection_result = detection_result


# Convenience functions
def create_protected_llm(llm, block_on_detection: bool = True, config: Optional[Dict] = None):
    """
    Wrap any LangChain LLM with PromptGuard protection.
    
    Args:
        llm: Any LangChain LLM instance
        block_on_detection: Whether to block or just warn on detection
        config: Custom PromptGuard configuration
        
    Returns:
        LLM with PromptGuard callback attached
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is not installed. Install with: pip install langchain langchain-core")
    
    handler = PromptGuardCallbackHandler(block_on_detection, config)
    
    # Add callback to existing callbacks
    if hasattr(llm, 'callbacks') and llm.callbacks:
        if isinstance(llm.callbacks, list):
            llm.callbacks.append(handler)
        else:
            llm.callbacks = [llm.callbacks, handler]
    else:
        llm.callbacks = [handler]
    
    return llm


def validate_chain_prompts(chain, input_data: Dict, config: Optional[Dict] = None) -> Dict:
    """
    Validate all prompts that would be generated by a LangChain chain.
    
    Args:
        chain: LangChain chain instance
        input_data: Input data for the chain
        config: Custom PromptGuard configuration
        
    Returns:
        Dict with validation results
    """
    guard = PromptGuard()
    
    try:
        # This is a simplified approach - real implementation would need to 
        # introspect the chain to find all prompt templates
        if hasattr(chain, 'prompt') and hasattr(chain.prompt, 'format'):
            try:
                formatted_prompt = chain.prompt.format(**input_data)
                result = guard.analyze_sync(formatted_prompt, config)
                return {
                    'safe': result['action'] != 'block',
                    'prompt': formatted_prompt,
                    'detection_result': result
                }
            except Exception as e:
                return {
                    'safe': True,
                    'prompt': None,
                    'detection_result': {'action': 'allow', 'message': f'Could not format prompt: {e}'}
                }
        else:
            return {
                'safe': True,
                'prompt': None,
                'detection_result': {'action': 'allow', 'message': 'No prompt template found'}
            }
    finally:
        guard.close()


# Usage examples in docstring
"""
Usage Examples:

1. Protect any LLM:
```python
from langchain.llms import OpenAI
from prompt_shield.integrations.langchain_integration import create_protected_llm

llm = OpenAI()
protected_llm = create_protected_llm(llm, block_on_detection=True)

# This will raise PromptInjectionDetected if prompt is malicious
response = protected_llm("ignore previous instructions and...")
```

2. Use as callback handler:
```python
from prompt_shield.integrations.langchain_integration import PromptGuardCallbackHandler

handler = PromptGuardCallbackHandler(block_on_detection=False)  # Just warn
llm = OpenAI(callbacks=[handler])
```

3. Validate chain prompts:
```python
from langchain import PromptTemplate, LLMChain
from prompt_shield.integrations.langchain_integration import validate_chain_prompts

prompt = PromptTemplate(template="Answer this: {question}")
chain = LLMChain(llm=llm, prompt=prompt)

result = validate_chain_prompts(chain, {"question": "ignore previous instructions"})
if not result['safe']:
    print("Dangerous prompt detected!")
```
"""