"""
PromptGuard integrations with popular AI frameworks.

Each integration is optional and only loads if the target framework is installed.
"""

# Try to import each integration, but don't fail if dependencies are missing
try:
    from .langchain import *
    __all__ = ['PromptGuardCallbackHandler', 'create_protected_llm', 'validate_chain_prompts']
except ImportError:
    pass

try:
    from .autogen import *
    if '__all__' in locals():
        __all__.extend(['PromptGuardAgent', 'protect_agent', 'create_protected_group_chat']) # type: ignore
    else:
        __all__ = ['PromptGuardAgent', 'protect_agent', 'create_protected_group_chat']
except ImportError:
    pass

