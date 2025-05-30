# Prompt Shield

Lightweight prompt injection detection and blocking for Python applications.

## Features

- **Fast Detection**: Keyword-based and semantic analysis
- **Configurable**: YAML-based rules and policies  
- **Lightweight**: Minimal dependencies, optional ML models
- **Multiple Layers**: Keywords, patterns, and semantic similarity
- **Easy Integration**: Simple API for any Python app

## Quick Start

```python
from prompt_shield import PromptGuard

guard = PromptGuard()

# Check a prompt
result = guard.analyze_sync("ignore previous instructions")
print(result['action'])  # 'block' or 'allow'
print(result['detected_hazards'])  # ['prompt_injection']

guard.close()
```

## Installation

### Basic (keyword detection only)
```bash
pip install prompt-shield
```

### Full (with semantic similarity)
```bash
pip install prompt-shield[full]
```

### Advanced (with pattern detection)
```bash
pip install prompt-shield[advanced]
```

## Configuration

### First time setup
```python
from prompt_shield.config_manager import init_user_config, edit_config

# Copy default config to ~/.prompt_shield/
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
- **Linux/Mac**: `~/.prompt_shield/config.yaml`
- **Windows**: `C:\Users\YourName\.prompt_shield\config.yaml`

## Detection Layers

1. **Keywords**: Fast regex-based detection from YAML patterns
2. **Semantic**: Embedding similarity (requires `[full]`)
3. **Advanced Patterns**: Unicode, obfuscation detection (requires `[advanced]`)

## Simple API

```python
from prompt_shield import is_safe, analyze

# Quick safety check
if is_safe("user input here"):
    process_safely(user_input)

# Detailed analysis  
result = analyze("user input here")
print(f"Action: {result['action']}")
print(f"Confidence: {result['confidence']}")
print(f"Hazards: {result['detected_hazards']}")
```

## License

MIT License - see LICENSE file for details.

## Contributing

Issues and pull requests welcome on GitHub!