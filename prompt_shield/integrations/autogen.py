"""
AutoGen integration for PromptGuard - Fixed Version
"""

from typing import Dict, Any, List, Optional, Union

try:
    from autogen import ConversableAgent, Agent  # type: ignore
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    # Create dummy base classes
    class ConversableAgent:  # type: ignore
        def __init__(self, **kwargs):
            pass
    Agent = ConversableAgent  # type: ignore

from ..detector import PromptGuard


class PromptGuardAgent(ConversableAgent):  # type: ignore
    """AutoGen agent with built-in prompt injection protection."""
    
    def __init__(self, 
                 name: str,
                 system_message: Optional[str] = None,
                 llm_config: Optional[Dict] = None,
                 guard_config: Optional[Dict] = None,
                 block_on_detection: bool = True,
                 **kwargs):
        """
        Initialize PromptGuard-protected AutoGen agent.
        
        Args:
            name: Agent name
            system_message: Agent system message
            llm_config: LLM configuration
            guard_config: PromptGuard configuration
            block_on_detection: Whether to block or warn on detection
            **kwargs: Other ConversableAgent arguments
        """
        if not AUTOGEN_AVAILABLE:
            raise ImportError("AutoGen is not installed. Install with: pip install pyautogen")
            
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs
        )
        
        self.guard = PromptGuard()
        self.guard_config = guard_config
        self.block_on_detection = block_on_detection
        
        # Override the message processing
        self._original_process_message = getattr(self, '_process_received_message', None)
        self._process_received_message = self._protected_process_message  # type: ignore
    
    def _protected_process_message(self, message: Union[Dict, str], sender, silent: bool = False) -> None:
        """Process message with prompt injection protection."""
        # Extract message content
        content = ""
        if isinstance(message, dict):
            content = message.get("content", "")
        elif isinstance(message, str):
            content = message
        
        # Validate the message
        if content:
            result = self.guard.analyze_sync(content, self.guard_config)
            
            if result['action'] == 'block':
                hazards = ', '.join(result['detected_hazards'])
                warning_msg = f"PromptGuard detected potential injection from {getattr(sender, 'name', 'unknown')}: {hazards}"
                
                if self.block_on_detection:
                    # Send warning back to sender instead of processing
                    try:
                        if hasattr(self, 'send') and hasattr(sender, 'receive'):
                            self.send(  # type: ignore
                                {
                                    "content": f"⚠️ Message blocked: {warning_msg}",
                                    "role": "assistant"
                                },
                                sender,
                                request_reply=False
                            )
                        else:
                            # If we can't send back to sender, just log the block
                            print(f"⚠️ Message blocked: {warning_msg}")
                    except Exception as e:
                        # If sending fails, just log the block
                        print(f"⚠️ Message blocked: {warning_msg} (failed to send warning: {e})")
                    return
                else:
                    print(f"⚠️ WARNING: {warning_msg}")
        
        # Process message normally if safe
        if self._original_process_message:
            # Try to call with the expected signature, handle gracefully if it fails
            try:
                # First try with silent parameter
                return self._original_process_message(message, sender, silent)
            except TypeError:
                # If that fails, try without silent parameter
                try:
                    return self._original_process_message(message, sender)
                except Exception as e:
                    # If both fail, just log and continue
                    print(f"Warning: Could not call original process_message: {e}")
                    return
    
    def send(self, 
             message: Union[Dict, str], 
             recipient,
             request_reply: Optional[bool] = None,
             silent: Optional[bool] = None) -> None:
        """Send message with validation."""
        # Validate outgoing message too
        content = ""
        if isinstance(message, dict):
            content = message.get("content", "")
        elif isinstance(message, str):
            content = message
        
        if content:
            result = self.guard.analyze_sync(content, self.guard_config)
            if result['action'] == 'block' and self.block_on_detection:
                hazards = ', '.join(result['detected_hazards'])
                agent_name = getattr(self, 'name', 'unknown')
                print(f"⚠️ Outgoing message blocked from {agent_name}: {hazards}")
                return
        
        return super().send(message, recipient, request_reply, silent)  # type: ignore
    
    def __del__(self):
        """Cleanup when agent is destroyed."""
        if hasattr(self, 'guard'):
            self.guard.close()


def protect_agent(agent, 
                 guard_config: Optional[Dict] = None,
                 block_on_detection: bool = True) -> ConversableAgent:
    """
    Add PromptGuard protection to an existing AutoGen agent.
    
    Args:
        agent: Existing AutoGen agent
        guard_config: PromptGuard configuration
        block_on_detection: Whether to block or warn on detection
        
    Returns:
        Protected agent (modifies original)
    """
    if not AUTOGEN_AVAILABLE:
        raise ImportError("AutoGen is not installed. Install with: pip install pyautogen")
    
    guard = PromptGuard()
    
    # Store original methods
    original_process = getattr(agent, '_process_received_message', None)
    original_send = agent.send
    
    def protected_process_message(message: Union[Dict, str], sender, **kwargs) -> None:
        """Protected message processing."""
        content = ""
        if isinstance(message, dict):
            content = message.get("content", "")
        elif isinstance(message, str):
            content = message
        
        if content:
            result = guard.analyze_sync(content, guard_config)
            if result['action'] == 'block':
                hazards = ', '.join(result['detected_hazards'])
                warning_msg = f"PromptGuard detected potential injection: {hazards}"
                
                if block_on_detection:
                    try:
                        if hasattr(agent, 'send') and hasattr(sender, 'receive'):
                            agent.send(
                                {"content": f"⚠️ Message blocked: {warning_msg}", "role": "assistant"},
                                sender,
                                request_reply=False
                            )
                        else:
                            print(f"⚠️ Message blocked: {warning_msg}")
                    except Exception as e:
                        print(f"⚠️ Message blocked: {warning_msg} (failed to send warning: {e})")
                    return
                else:
                    print(f"⚠️ WARNING: {warning_msg}")
        
        # Call original method if it exists
        if original_process:
            return original_process(message, sender, **kwargs)
    
    def protected_send(message: Union[Dict, str], 
                      recipient,
                      request_reply: Optional[bool] = None,
                      silent: Optional[bool] = None) -> None:
        """Protected send method."""
        content = ""
        if isinstance(message, dict):
            content = message.get("content", "")
        elif isinstance(message, str):
            content = message
        
        if content:
            result = guard.analyze_sync(content, guard_config)
            if result['action'] == 'block' and block_on_detection:
                hazards = ', '.join(result['detected_hazards'])
                print(f"⚠️ Outgoing message blocked: {hazards}")
                return
        
        return original_send(message, recipient, request_reply, silent)
    
    # Apply protection
    agent._process_received_message = protected_process_message
    agent.send = protected_send
    agent._prompt_guard = guard  # Store reference for cleanup
    
    return agent


def create_protected_group_chat(agents: List, 
                               guard_config: Optional[Dict] = None,
                               block_on_detection: bool = True) -> List:
    """
    Create a group chat with all agents protected.
    
    Args:
        agents: List of AutoGen agents
        guard_config: PromptGuard configuration
        block_on_detection: Whether to block or warn on detection
        
    Returns:
        List of protected agents
    """
    return [protect_agent(agent, guard_config, block_on_detection) for agent in agents]


class PromptInjectionDetected(Exception):
    """Exception raised when prompt injection is detected."""
    
    def __init__(self, message: str, detection_result: Dict):
        super().__init__(message)
        self.detection_result = detection_result