import shutil
import yaml
from pathlib import Path

def get_config_path():
    """Get the user's config file path."""
    return Path.home() / ".llm_prompt_shield" / "config.yaml"

def get_data_path(filename):
    """Get the user's data file path."""
    return Path.home() / ".llm_prompt_shield" / "data" / filename

def init_user_config():
    """Initialize user config directory with default files."""
    user_config_dir = Path.home() / ".llm_prompt_shield"
    user_data_dir = user_config_dir / "data"
    
    # Create directories
    user_config_dir.mkdir(exist_ok=True)
    user_data_dir.mkdir(exist_ok=True)
    
    # Copy default files if they don't exist
    package_dir = Path(__file__).parent
    
    config_files = [
        ("config.yaml", user_config_dir / "config.yaml"),
        ("data/keywords.yaml", user_data_dir / "keywords.yaml"),
        ("data/semantic_examples.yaml", user_data_dir / "semantic_examples.yaml")
    ]
    
    for src_rel, dst in config_files:
        src = package_dir / src_rel
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"Created {dst}")
    
    return user_config_dir

def load_user_config():
    """Load config from user directory, falling back to package defaults."""
    user_config_path = get_config_path()
    
    if user_config_path.exists():
        with open(user_config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        # Fall back to package default
        package_config = Path(__file__).parent / "config.yaml"
        if package_config.exists():
            with open(package_config, "r") as f:
                return yaml.safe_load(f)
    
    # Ultimate fallback
    return {
        "hazard_policies": {
            "prompt_injection": "block",
            "data_extraction": "block",
            "default": "allow"
        },
        "semantic_threshold": 0.5,
        "debug_logging": False
    }

def edit_config():
    """Open the user config file for editing."""
    config_path = get_config_path()
    
    if not config_path.exists():
        print("Initializing user config...")
        init_user_config()
    
    print(f"Edit your config file at: {config_path}")
    
    # Try to open with default editor
    import subprocess
    import sys
    
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", str(config_path)])
        elif sys.platform == "win32":  # Windows
            subprocess.run(["notepad", str(config_path)])
        else:  # Linux
            subprocess.run(["xdg-open", str(config_path)])
    except:
        print(f"Please manually edit: {config_path}")

def reset_config():
    """Reset user config to defaults."""
    user_config_dir = Path.home() / ".llm_prompt_shield"
    if user_config_dir.exists():
        shutil.rmtree(user_config_dir)
    init_user_config()
    print("Config reset to defaults")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "init":
            init_user_config()
        elif command == "edit":
            edit_config()
        elif command == "reset":
            reset_config()
        else:
            print("Usage: python -m llm_prompt_shield.config_manager [init|edit|reset]")
    else:
        edit_config()