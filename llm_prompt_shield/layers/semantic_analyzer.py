from __future__ import annotations

import os
import re
import joblib
import yaml
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

# Initialize globals with None to handle import errors gracefully
_MODEL = None
_hazard_examples = {}
_keyword_patterns = {}
_config = {}
_keyword_regexes = {}
_hazard_embeddings = {}

def _load_sentence_transformer():
    """Load sentence transformer with error handling."""
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer("all-mpnet-base-v2")
        except ImportError:
            print("Warning: sentence-transformers not available, using keyword-only detection")
            _MODEL = False  # Set to False to indicate unavailable
    return _MODEL

def _load_yaml(filename: str) -> Dict[str, List[str]]:
    """Load YAML file with error handling."""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "../data")
        path = os.path.join(data_dir, filename)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file {filename} not found at {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return {k.lower(): v for k, v in data.items()}
    except Exception as e:
        raise RuntimeError(f"Could not load required file {filename}: {e}")

def _load_config() -> Dict:
    """Load configuration including semantic threshold."""
    try:
        # Try user config first
        from llm_prompt_shield.user_config import load_user_config
        return load_user_config()
    except Exception:
        # Fall back to package config
        try:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            if not os.path.exists(config_path):
                return {"semantic_threshold": 0.5, "debug_logging": False}
                
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                # Ensure we have defaults
                config.setdefault("semantic_threshold", 0.5)
                config.setdefault("debug_logging", False)
                return config
        except Exception:
            return {"semantic_threshold": 0.5, "debug_logging": False}

    # Initialize data
def _initialize_data():
    """Initialize all data structures."""
    global _hazard_examples, _keyword_patterns, _config, _keyword_regexes, _hazard_embeddings
    
    _hazard_examples = _load_yaml("pattern_detector.yaml")
    _keyword_patterns = _load_yaml("patterns.yaml")
    _config = _load_config()
    
    # Set up debug printing based on config
    debug_enabled = _config.get("debug_logging", False)
    
    def debug_print(*args, **kwargs):
        if debug_enabled:
            print(*args, **kwargs)
    
    # Compile keyword patterns - now they should be proper regex
    _keyword_regexes = {}
    for label, patterns in _keyword_patterns.items():
        compiled_patterns = []
        for pattern in patterns:
            try:
                compiled_patterns.append(re.compile(pattern, re.I | re.DOTALL))
                debug_print(f"✅ Compiled pattern: {pattern}")
            except Exception as e:
                debug_print(f"❌ Error compiling pattern '{pattern}': {e}")
                continue
        _keyword_regexes[label] = compiled_patterns
        debug_print(f"📝 Loaded {len(compiled_patterns)} patterns for {label}")
    
    # Initialize embeddings
    _hazard_embeddings = _load_or_build_embeddings()

def _load_or_build_embeddings() -> Dict[str, np.ndarray]:
    """Load embeddings from cache or build new ones."""
    cache_path = _get_cache_path()
    
    # Try to load from cache
    if cache_path.exists():
        try:
            cached_embeddings = joblib.load(cache_path)
            # Validate cache compatibility
            test_emb = _safe_encode(["test"])
            if test_emb.size > 0:
                sample_emb = next(iter(cached_embeddings.values()), None)
                if sample_emb is not None and sample_emb.shape[1] == test_emb.shape[1]:
                    return cached_embeddings
        except Exception:
            pass  # Cache invalid, will rebuild
    
    # Build new embeddings
    embeddings = _build_embeddings()
    
    # Try to save to cache (but don't fail if we can't)
    if embeddings:
        try:
            # Ensure cache directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(embeddings, cache_path)
        except Exception:
            pass  # Failed to cache, but embeddings still work
    
    return embeddings

# Vector utilities
def _safe_encode(texts: List[str]) -> np.ndarray:
    """Safely encode texts to normalized embeddings."""
    model = _load_sentence_transformer()
    if not model or model is False:
        # Return empty array if model unavailable
        return np.empty((0, 768), dtype=np.float32)
    
    # Filter out empty or invalid texts
    valid_texts = [str(text).strip() for text in texts if text and str(text).strip()]
    if not valid_texts:
        embedding_dim = getattr(model, 'get_sentence_embedding_dimension', lambda: 768)()
        return np.empty((0, embedding_dim), dtype=np.float32)
    
    try:
        enc = model.encode(valid_texts, convert_to_numpy=True)
        
        # Handle NaN, inf values
        enc = np.nan_to_num(enc, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize vectors
        norms = np.linalg.norm(enc, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms < 1e-12, 1.0, norms)
        enc_normalized = enc / norms
        
        # Final validation
        enc_normalized = np.nan_to_num(enc_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return enc_normalized
        
    except Exception:
        model = _load_sentence_transformer()
        embedding_dim = getattr(model, 'get_sentence_embedding_dimension', lambda: 768)()
        return np.empty((0, embedding_dim), dtype=np.float32)

def _is_valid_embedding(vec: np.ndarray) -> bool:
    """Check if embedding is valid for similarity computation."""
    if not isinstance(vec, np.ndarray):
        return False
    if vec.size == 0:
        return False
    if not np.isfinite(vec).all():
        return False
    # Check if all values are zero (invalid embedding)
    if np.allclose(vec, 0.0, atol=1e-12):
        return False
    return True

def _get_cache_path():
    """Get cache file path in user directory.""" 
    try:
        from llm_prompt_shield.user_config import get_config_path
        user_config_dir = get_config_path().parent
        return user_config_dir / "hazard_embeddings.joblib"
    except:
        # Fall back to temp directory if config manager not available
        import tempfile
        return Path(tempfile.gettempdir()) / "llm_prompt_shield_embeddings.joblib"

def _build_embeddings() -> Dict[str, np.ndarray]:
    """Build embeddings for all hazard examples."""
    model = _load_sentence_transformer()
    if not model or model is False:
        return {}
    
    out = {}
    for label, examples in _hazard_examples.items():
        if not examples:
            continue
            
        try:
            emb = _safe_encode(examples)
            if emb.size > 0 and _is_valid_embedding(emb):
                out[label] = emb
        except Exception:
            continue
    return out

def _safe_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity with safety checks, avoiding sklearn issues."""
    try:
        # Additional validation
        if not (_is_valid_embedding(a) and _is_valid_embedding(b)):
            return 0.0
            
        # Ensure both arrays are 2D
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
            
        # Check dimensions match
        if a.shape[1] != b.shape[1]:
            return 0.0
        
        # Manual cosine similarity computation to avoid sklearn issues
        max_similarity = 0.0
        
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                vec_a = a[i]
                vec_b = b[j]
                
                # Compute dot product
                dot_product = np.dot(vec_a, vec_b)
                
                # Compute norms
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)
                
                # Check for zero norms (should not happen with our normalization, but be safe)
                if norm_a < 1e-12 or norm_b < 1e-12:
                    similarity = 0.0
                else:
                    similarity = dot_product / (norm_a * norm_b)
                
                # Handle any remaining numerical issues
                if not np.isfinite(similarity):
                    similarity = 0.0
                else:
                    # Clamp to valid range
                    similarity = max(-1.0, min(1.0, similarity))
                
                max_similarity = max(max_similarity, similarity)
        
        return float(max_similarity)
        
    except Exception:
        return 0.0

def classify_hazards(prompt: str, threshold: Optional[float] = None) -> List[str]:
    """Classify prompt for potential hazards using keywords and semantic similarity."""
    # Lazy initialization
    if not _keyword_regexes:
        _initialize_data()
    
    if not prompt or not str(prompt).strip():
        return []
    
    # Get threshold from config or use default
    if threshold is None:
        config_threshold = _config.get('semantic_threshold', 0.5)
        threshold = float(config_threshold) if isinstance(config_threshold, (int, float)) else 0.5
        
    prompt = str(prompt).strip()
    matched = set()

    # Check if debug logging is enabled
    debug_enabled = _config.get("debug_logging", False)
    
    def debug_print(*args, **kwargs):
        if debug_enabled:
            print(*args, **kwargs)

    debug_print(f"🔍 Analyzing prompt: '{prompt}'")

    # Layer 1: keyword matches using YAML patterns
    for label, regexes in _keyword_regexes.items():
        for regex in regexes:
            try:
                if regex.search(prompt):
                    matched.add(label)
                    debug_print(f"🎯 Pattern match for {label}: {regex.pattern[:50]}...")
                    break  # Found a match for this label, move to next label
            except Exception as e:
                debug_print(f"❌ Error matching pattern {regex.pattern}: {e}")
                continue

    debug_print(f"📊 Keyword layer results: {list(matched)}")

    # Layer 2: semantic similarity (only if sentence transformers available)
    model = _load_sentence_transformer()
    if not model or model is False or not _hazard_embeddings:
        debug_print("⚠️  Semantic similarity disabled (no model or embeddings)")
        final_result = list(matched)
        debug_print(f"🏁 Final classification: {final_result}")
        return final_result
        
    try:
        prompt_emb = _safe_encode([prompt])
        if not _is_valid_embedding(prompt_emb):
            debug_print("⚠️  Could not create valid embedding for prompt")
            final_result = list(matched)
            debug_print(f"🏁 Final classification: {final_result}")
            return final_result
    except Exception as e:
        debug_print(f"⚠️  Error creating embedding: {e}")
        final_result = list(matched)
        debug_print(f"🏁 Final classification: {final_result}")
        return final_result

    for label, ref_emb in _hazard_embeddings.items():
        try:
            max_score = _safe_cosine_similarity(prompt_emb, ref_emb)
            debug_print(f"📈 Semantic similarity for {label}: {max_score:.3f} (threshold: {threshold})")
            if max_score >= threshold:
                matched.add(label)
                debug_print(f"🎯 Semantic match found for {label}")
        except Exception as e:
            debug_print(f"❌ Error in semantic similarity for {label}: {e}")
            continue

    final_result = list(matched)
    debug_print(f"🏁 Final classification: {final_result}")
    return final_result

# Initialize data on import
_initialize_data()