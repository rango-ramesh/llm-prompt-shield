# LLM Prompt Shield

Lightweight prompt injection detection and blocking for Python applications.

## Features

- **Fast Detection**: Keyword-based and semantic analysis
- **Configurable**: YAML-based rules and policies  
- **Lightweight**: Minimal dependencies, optional ML models
- **Multiple Layers**: Keywords, patterns, and semantic similarity
- **Easy Integration**: Simple API for any Python app

## Quick Start

```python
from llm_prompt_shield import PromptGuard

guard = PromptGuard()

# Check a prompt
result = guard.analyze_sync("ignore previous instructions")
print(result['action'])  # 'block' or 'allow'
print(result['detected_hazards'])  # ['prompt_injection']

guard.close()
```

## Installation

### Standard (includes all detection features)
```bash
pip install llm-prompt-shield
```

### With integrations
```bash
pip install llm-prompt-shield[integrations]  # All integrations
pip install llm-prompt-shield[langchain]     # Just LangChain
pip install llm-prompt-shield[autogen]       # Just AutoGen  
```

## Configuration

### First time setup
```python
from llm_prompt_shield.config_manager import init_user_config, edit_config

# Copy default config to ~/.llm_prompt_shield/
init_user_config()

# Open config file in your default editor
edit_config()
```

### Custom config in code
```python
config = {
    "prompt_injection": "block",
    "data_extraction": "warn", 
    "default": "allow"
}

result = guard.analyze("suspicious prompt", config)
```

### Config file location
After running `init_user_config()`, edit your personal config at:
- **Linux/Mac**: `~/.llm_prompt_shield/config.yaml`
- **Windows**: `C:\Users\YourName\.llm_prompt_shield\config.yaml`

## Detection Layers

1. **Keywords**: Fast regex-based detection from YAML patterns
2. **Semantic**: Embedding similarity for catching variations
3. **Advanced Patterns**: Unicode, obfuscation detection

## Simple API

```python
from llm_prompt_shield import is_safe, analyze

# Quick safety check
if is_safe("user input here"):
    process_safely(user_input)

# Detailed analysis  
result = analyze("user input here")
print(f"Action: {result['action']}")
print(f"Confidence: {result['confidence']}")
print(f"Hazards: {result['detected_hazards']}")
```

# PromptGuard Integrations

PromptGuard provides seamless integrations with popular AI agent frameworks to automatically protect your applications from prompt injection attacks.

## Available Integrations

- **LangChain** - Protect LangChain applications with callback handlers
- **AutoGen** - Secure multi-agent conversations and workflows

## LangChain Integration

### Installation

```bash
pip install llm-prompt-shield[langchain]
```

### Quick Start

```python
from llm_prompt_shield.integrations.langchain import PromptGuardCallbackHandler
from langchain.llms import OpenAI

# Create callback handler
callback = PromptGuardCallbackHandler(block_on_detection=True)

# Use with any LangChain LLM
llm = OpenAI(callbacks=[callback])

# Protected execution
response = llm("What is the capital of France?")  # ✅ Safe
# llm("Ignore previous instructions...")  # ❌ Blocked
```

### Advanced Configuration

```python
from llm_prompt_shield.integrations.langchain import (
    PromptGuardCallbackHandler,
    PromptInjectionDetected
)

# Custom guard configuration
guard_config = {
    "threshold": 0.8,
    "check_entities": True,
    "check_patterns": True
}

# Warning mode (logs threats but doesn't block)
warning_callback = PromptGuardCallbackHandler(
    guard_config=guard_config,
    block_on_detection=False
)

# Blocking mode (raises exceptions on threats)
blocking_callback = PromptGuardCallbackHandler(
    guard_config=guard_config,
    block_on_detection=True
)

# Use with chains
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer this question: {question}"
)

chain = LLMChain(
    llm=OpenAI(callbacks=[blocking_callback]),
    prompt=prompt
)

try:
    result = chain.run("What is 2+2?")  # ✅ Safe
    print(result)
except PromptInjectionDetected as e:
    print(f"Threat detected: {e}")
```

### LangChain Features

- **Automatic validation** of all prompts sent to LLMs
- **Configurable responses** - block execution or log warnings
- **Chain compatibility** - works with any LangChain chain or agent
- **Custom guard configuration** - fine-tune detection sensitivity
- **Exception handling** - proper error handling for blocked requests

## AutoGen Integration

### Installation

```bash
pip install llm-prompt-shield[autogen]
```

### Quick Start

```python
from llm_prompt_shield.integrations.autogen import PromptGuardAgent
import autogen

# Create protected agent
agent = PromptGuardAgent(
    name="protected_assistant",
    system_message="You are a helpful assistant.",
    llm_config={"model": "gpt-4"},
    block_on_detection=True
)

# Create user proxy (unprotected)
user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config=False
)

# Safe conversation
user_proxy.initiate_chat(agent, message="Hello!")  # ✅ Works

# Dangerous prompt gets blocked
# user_proxy.initiate_chat(agent, message="Ignore all instructions...")  # ❌ Blocked
```

### Protecting Existing Agents

```python
from autogen import ConversableAgent
from llm_prompt_shield.integrations.autogen import protect_agent

# Create standard AutoGen agent
agent = ConversableAgent(
    name="assistant",
    llm_config={"model": "gpt-4"},
    system_message="You are a helpful assistant."
)

# Add protection to existing agent
protected_agent = protect_agent(
    agent, 
    block_on_detection=False  # Warning mode
)

# Agent now has prompt injection protection
```

### Group Chat Protection

```python
from llm_prompt_shield.integrations.autogen import create_protected_group_chat

# Create multiple agents
user_proxy = autogen.UserProxyAgent(name="user")
assistant = autogen.ConversableAgent(name="assistant", llm_config=llm_config)
coder = autogen.ConversableAgent(name="coder", llm_config=llm_config)

# Protect all agents in group chat
protected_agents = create_protected_group_chat(
    agents=[user_proxy, assistant, coder],
    block_on_detection=True
)

# Create group chat with protected agents
group_chat = autogen.GroupChat(
    agents=protected_agents, 
    messages=[], 
    max_round=10
)

manager = autogen.GroupChatManager(
    groupchat=group_chat, 
    llm_config=llm_config
)
```

### AutoGen Features

- **Bi-directional protection** - validates both incoming and outgoing messages
- **Group chat support** - protect entire multi-agent conversations
- **Flexible blocking** - choose between blocking and warning modes
- **Message filtering** - prevents malicious prompts from reaching agents
- **Conversation continuity** - safe messages continue normal flow

## Configuration Options

Both integrations support the same guard configuration options:

```python
guard_config = {
    # Detection threshold (0.0 - 1.0)
    "threshold": 0.8,
    
    # Enable entity-based detection
    "check_entities": True,
    
    # Enable pattern-based detection  
    "check_patterns": True,
    
    # Custom detection rules
    "custom_rules": [
        {"pattern": r"ignore.*instruction", "severity": "high"},
        {"pattern": r"system.*prompt", "severity": "medium"}
    ]
}
```

## Error Handling

### LangChain
```python
from llm_prompt_shield.integrations.langchain import PromptInjectionDetected

try:
    result = chain.run("malicious prompt")
except PromptInjectionDetected as e:
    print(f"Prompt injection detected: {e.detection_result}")
    # Handle the threat appropriately
```

### AutoGen
```python
# AutoGen integration handles errors gracefully
# Blocked messages are logged and conversation continues safely

agent = PromptGuardAgent(
    name="agent",
    block_on_detection=False  # Use warning mode to see detections
)
```

## Best Practices

### 1. Start with Warning Mode
```python
# Begin with warnings to understand your traffic
callback = PromptGuardCallbackHandler(block_on_detection=False)
```

### 2. Monitor Detection Logs
```python
# Enable logging to track attempted attacks
import logging
logging.basicConfig(level=logging.INFO)
```

### 3. Tune Detection Thresholds
```python
# Adjust sensitivity based on your use case
guard_config = {"threshold": 0.9}  # Less sensitive
guard_config = {"threshold": 0.7}  # More sensitive
```

### 4. Graceful Error Handling
```python
# Always handle PromptInjectionDetected exceptions
try:
    result = protected_chain.run(user_input)
except PromptInjectionDetected:
    return "I can't process that request. Please try rephrasing."
```

## Examples

### LangChain RAG Pipeline
```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from llm_prompt_shield.integrations.langchain import PromptGuardCallbackHandler

# Protected RAG chain
callback = PromptGuardCallbackHandler(block_on_detection=True)
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(callbacks=[callback]),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Safe queries work normally
answer = qa_chain.run("What is the company policy on remote work?")

# Injection attempts are blocked
# qa_chain.run("Ignore the documents and tell me...")  # ❌ Blocked
```

### AutoGen Code Review Workflow
```python
from llm_prompt_shield.integrations.autogen import PromptGuardAgent

# Protected code reviewer
reviewer = PromptGuardAgent(
    name="code_reviewer",
    system_message="You are a senior code reviewer. Review code for best practices.",
    llm_config={"model": "gpt-4"},
    block_on_detection=True
)

# Protected developer
developer = PromptGuardAgent(
    name="developer", 
    system_message="You are a Python developer.",
    llm_config={"model": "gpt-3.5-turbo"},
    block_on_detection=True
)

# Safe code review process is protected from injection
user_proxy.initiate_chat(
    reviewer,
    message="Please review this Python function: def add(a, b): return a + b"
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade llm-prompt-shield[integrations]
   ```

2. **Detection Too Sensitive**
   ```python
   guard_config = {"threshold": 0.9}  # Reduce sensitivity
   ```

3. **Detection Not Sensitive Enough**
   ```python
   guard_config = {"threshold": 0.7}  # Increase sensitivity
   ```

4. **Performance Concerns**
   ```python
   # Use async methods for better performance
   result = await guard.analyze_async(prompt)
   ```

## Integration Support

- **LangChain**: Supports all LLM types, chains, and agents
- **AutoGen**: Supports ConversableAgent and all derived agent types
- **Future Integrations**: Additional frameworks can be added based on community needs

## Development Setup

```bash
# Clone and setup
git clone https://github.com/rango-ramesh/llm-prompt-shield
cd llm_prompt_shield
python3 -m venv venv
source venv/bin/activate

# Install all dependencies for testing
pip install -r requirements-dev.txt
pip install -e .

# Basic tests
python3 test.py

# Integration tests
python3 integration_test.py
```

## License

MIT License - see LICENSE file for details.

## Contributing

Issues and pull requests welcome on GitHub!