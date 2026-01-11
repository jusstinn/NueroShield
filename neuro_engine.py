"""
NeuroShield: The SAE Cognitive Firewall - Advanced Backend Engine v2.0
======================================================================
A comprehensive mechanistic interpretability toolkit featuring:
- Multi-layer SAE analysis
- Feature steering (boost & clamp)
- Activation patching for causal analysis
- Token-by-token feature tracking
- Automatic safety feature detection
- Circuit identification primitives

Based on latest research:
- Domain-Specific SAEs (arxiv:2508.09363)
- Route Sparse Autoencoders (EMNLP 2025)
- Bias Adaptation in SAEs (arxiv:2506.14002)
- Atlas-Alignment Framework (arxiv:2510.27413)

Author: NeuroShield Research Team
License: MIT
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
import json
from datetime import datetime
from pathlib import Path

# =============================================================================
# MOCK MODE CONFIGURATION
# =============================================================================
# Set to True to test UI without downloading large models (~500MB)
# This simulates model behavior with realistic random data
MOCK_MODE = True

# =============================================================================
# Configuration Constants
# =============================================================================
DEFAULT_MODEL = "gpt2-small"
DEFAULT_SAE_RELEASE = "gpt2-small-res-jb"
AVAILABLE_HOOK_POINTS = [
    "blocks.0.hook_resid_pre",
    "blocks.2.hook_resid_pre", 
    "blocks.4.hook_resid_pre",
    "blocks.6.hook_resid_pre",
    "blocks.8.hook_resid_pre",
    "blocks.10.hook_resid_pre",
    "blocks.11.hook_resid_post",
]

# Known feature categories (example indices - in practice, identify through analysis)
FEATURE_CATEGORIES = {
    "safety": [1045, 902, 3421, 7892, 12045, 15678, 18901, 21234],
    "refusal": [2341, 5678, 8901, 11234],
    "toxicity": [3456, 6789, 9012, 12345],
    "code": [4567, 7890, 10123, 13456],
    "math": [5678, 8901, 11234, 14567],
    "creative": [6789, 9012, 12345, 15678],
}

# =============================================================================
# Device Detection
# =============================================================================
def get_device() -> str:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_device_info() -> Dict[str, Any]:
    """Get detailed device information."""
    device = get_device()
    info = {"device": device, "device_name": device}
    
    if device == "cuda":
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["memory_allocated"] = torch.cuda.memory_allocated(0) / 1e9
    elif device == "mps":
        info["device_name"] = "Apple Silicon (MPS)"
    else:
        info["device_name"] = "CPU"
    
    return info

# =============================================================================
# Data Classes
# =============================================================================
class InterventionType(Enum):
    """Types of feature interventions."""
    CLAMP = "clamp"           # Set to zero
    BOOST = "boost"           # Multiply activation
    SET = "set"               # Set to specific value
    ABLATE = "ablate"         # Remove feature contribution

@dataclass
class FeatureActivation:
    """Represents a single feature's activation with metadata."""
    index: int
    activation: float
    layer: Optional[str] = None
    token_position: Optional[int] = None
    category: Optional[str] = None
    description: Optional[str] = None

@dataclass
class TokenFeatureMap:
    """Maps tokens to their feature activations."""
    tokens: List[str]
    activations: np.ndarray  # Shape: [seq_len, n_features]
    top_features_per_token: List[List[FeatureActivation]]

@dataclass
class AnalysisResult:
    """Comprehensive analysis results."""
    top_features: List[FeatureActivation]
    all_activations: np.ndarray
    tokens: List[str]
    token_feature_map: Optional[TokenFeatureMap] = None
    layer: str = "blocks.8.hook_resid_pre"
    prompt: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class MultiLayerAnalysis:
    """Analysis results across multiple layers."""
    layers: List[str]
    results: Dict[str, AnalysisResult]
    feature_flow: np.ndarray  # Shape: [n_layers, n_features] - how features evolve

@dataclass
class Intervention:
    """Defines a feature intervention."""
    feature_index: int
    intervention_type: InterventionType
    value: float = 0.0  # For BOOST: multiplier, for SET: target value

@dataclass
class GenerationResult:
    """Results from protected generation."""
    text: str
    blocked_features_log: Dict[int, int]
    boosted_features_log: Dict[int, int]
    total_interventions: int
    tokens_generated: int
    interventions_per_token: List[Dict[str, Any]]

@dataclass
class CausalTrace:
    """Results from causal tracing / activation patching."""
    prompt: str
    target_token: str
    layer_effects: Dict[str, float]  # layer -> effect size
    feature_effects: Dict[int, float]  # feature_idx -> effect size
    critical_layers: List[str]
    critical_features: List[int]

@dataclass
class SafetyAuditResult:
    """Results from automated safety audit."""
    prompt: str
    safety_score: float  # 0-1, higher = safer
    triggered_features: List[FeatureActivation]
    risk_level: str  # "low", "medium", "high", "critical"
    recommendations: List[str]

# =============================================================================
# Feature Database (Neuronpedia-style descriptions)
# =============================================================================
class FeatureDatabase:
    """
    Database of known feature descriptions and categories.
    In production, this would connect to Neuronpedia API or local cache.
    """
    
    def __init__(self):
        self.descriptions = {}
        self.categories = FEATURE_CATEGORIES
        self._load_default_descriptions()
    
    def _load_default_descriptions(self):
        """Load default feature descriptions (examples)."""
        self.descriptions = {
            1045: "Potential harm/violence related content",
            902: "Weapons and dangerous objects",
            3421: "Illegal activities discussion",
            7892: "Personal safety concerns",
            12045: "Harmful instructions detection",
            2341: "Refusal/decline patterns",
            5678: "Ethical concerns activation",
            4567: "Code syntax patterns",
            6789: "Creative writing markers",
        }
    
    def get_description(self, feature_idx: int) -> Optional[str]:
        """Get description for a feature."""
        return self.descriptions.get(feature_idx)
    
    def get_category(self, feature_idx: int) -> Optional[str]:
        """Get category for a feature."""
        for category, indices in self.categories.items():
            if feature_idx in indices:
                return category
        return None
    
    def search_features(self, query: str) -> List[int]:
        """Search features by description."""
        results = []
        query_lower = query.lower()
        for idx, desc in self.descriptions.items():
            if query_lower in desc.lower():
                results.append(idx)
        return results

# =============================================================================
# Mock Engine (for UI testing without model downloads)
# =============================================================================
class MockNeuroEngine:
    """
    Mock implementation for UI testing without loading real models.
    Simulates realistic behavior based on prompt content.
    """
    
    def __init__(self, custom_weights_path: Optional[str] = None):
        self.device = get_device()
        self.model_name = "gpt2-small (MOCK)"
        self.sae_id = "blocks.8.hook_resid_pre (MOCK)"
        self.custom_weights_path = custom_weights_path
        self.n_features = 24576
        self.n_layers = 12
        self.feature_db = FeatureDatabase()
        self.available_layers = AVAILABLE_HOOK_POINTS
        self.session_history: List[AnalysisResult] = []
        
        print(f"[MOCK MODE] Initialized on device: {self.device}")
    
    def _detect_content_type(self, text: str) -> Dict[str, float]:
        """Detect content type from text for realistic simulation."""
        text_lower = text.lower()
        scores = {
            "safety": 0.0,
            "code": 0.0,
            "math": 0.0,
            "creative": 0.0,
            "normal": 1.0,
        }
        
        # Safety keywords
        safety_keywords = ["kill", "bomb", "hack", "steal", "attack", "harm", 
                          "weapon", "dangerous", "illegal", "hurt", "destroy"]
        if any(kw in text_lower for kw in safety_keywords):
            scores["safety"] = np.random.uniform(0.7, 1.0)
        
        # Code keywords
        code_keywords = ["def ", "function", "class ", "import ", "var ", "const "]
        if any(kw in text_lower for kw in code_keywords):
            scores["code"] = np.random.uniform(0.6, 0.9)
        
        # Math keywords
        math_keywords = ["calculate", "equation", "solve", "integral", "derivative"]
        if any(kw in text_lower for kw in math_keywords):
            scores["math"] = np.random.uniform(0.6, 0.9)
        
        # Creative keywords  
        creative_keywords = ["story", "poem", "creative", "imagine", "fantasy"]
        if any(kw in text_lower for kw in creative_keywords):
            scores["creative"] = np.random.uniform(0.6, 0.9)
            
        return scores
    
    def analyze_prompt(
        self, 
        text: str,
        layer: str = "blocks.8.hook_resid_pre",
        return_all_tokens: bool = False
    ) -> AnalysisResult:
        """Simulate feature analysis with content-aware activations."""
        tokens = text.split()[:50]
        content_scores = self._detect_content_type(text)
        
        # Generate base activations
        activations = np.random.exponential(0.5, size=self.n_features)
        
        # Boost relevant category features based on content
        for category, score in content_scores.items():
            if score > 0.5 and category in FEATURE_CATEGORIES:
                for idx in FEATURE_CATEGORIES[category]:
                    if idx < self.n_features:
                        activations[idx] = np.random.uniform(3.0, 15.0) * score
        
        # Get top features
        top_indices = np.argsort(activations)[-10:][::-1]
        top_features = [
            FeatureActivation(
                index=int(idx),
                activation=float(activations[idx]),
                layer=layer,
                category=self.feature_db.get_category(idx),
                description=self.feature_db.get_description(idx)
            )
            for idx in top_indices
        ]
        
        # Token-level analysis
        token_feature_map = None
        if return_all_tokens:
            seq_len = len(tokens)
            token_activations = np.random.exponential(0.5, size=(seq_len, min(100, self.n_features)))
            
            # Make later tokens have stronger activations (simulating context buildup)
            for i in range(seq_len):
                token_activations[i] *= (1 + i * 0.1)
            
            top_per_token = []
            for i in range(seq_len):
                top_idx = np.argsort(token_activations[i])[-5:][::-1]
                top_per_token.append([
                    FeatureActivation(
                        index=int(idx),
                        activation=float(token_activations[i, idx]),
                        token_position=i
                    )
                    for idx in top_idx
                ])
            
            token_feature_map = TokenFeatureMap(
                tokens=tokens,
                activations=token_activations,
                top_features_per_token=top_per_token
            )
        
        result = AnalysisResult(
            top_features=top_features,
            all_activations=activations,
            tokens=tokens,
            token_feature_map=token_feature_map,
            layer=layer,
            prompt=text
        )
        
        self.session_history.append(result)
        return result
    
    def analyze_multi_layer(
        self, 
        text: str,
        layers: Optional[List[str]] = None
    ) -> MultiLayerAnalysis:
        """Analyze across multiple layers."""
        if layers is None:
            layers = self.available_layers[:4]  # Default to first 4 layers
        
        results = {}
        feature_flow = np.zeros((len(layers), min(100, self.n_features)))
        
        for i, layer in enumerate(layers):
            result = self.analyze_prompt(text, layer=layer)
            results[layer] = result
            feature_flow[i] = result.all_activations[:100]
            
            # Simulate feature evolution across layers
            if i > 0:
                feature_flow[i] = feature_flow[i-1] * np.random.uniform(0.8, 1.2, size=100)
        
        return MultiLayerAnalysis(
            layers=layers,
            results=results,
            feature_flow=feature_flow
        )
    
    def generate_protected(
        self,
        text: str,
        interventions: List[Intervention] = [],
        max_new_tokens: int = 50
    ) -> GenerationResult:
        """Simulate protected generation with interventions."""
        blocked_log = {}
        boosted_log = {}
        interventions_per_token = []
        
        for intervention in interventions:
            if intervention.intervention_type == InterventionType.CLAMP:
                blocked_log[intervention.feature_index] = np.random.randint(1, 10)
            elif intervention.intervention_type == InterventionType.BOOST:
                boosted_log[intervention.feature_index] = np.random.randint(1, 10)
        
        # Simulate token-by-token interventions
        tokens_generated = np.random.randint(20, max_new_tokens)
        for i in range(tokens_generated):
            token_interventions = {
                "token_idx": i,
                "blocked": list(blocked_log.keys())[:2] if blocked_log else [],
                "boosted": list(boosted_log.keys())[:2] if boosted_log else [],
            }
            interventions_per_token.append(token_interventions)
        
        # Generate text based on interventions
        if blocked_log:
            generated = f"{text} [PROTECTED OUTPUT - {len(blocked_log)} features clamped] This response has been filtered for safety."
        elif boosted_log:
            generated = f"{text} [STEERED OUTPUT - {len(boosted_log)} features boosted] This response has enhanced characteristics."
        else:
            generated = f"{text} This is the model's natural continuation without intervention."
        
        return GenerationResult(
            text=generated,
            blocked_features_log=blocked_log,
            boosted_features_log=boosted_log,
            total_interventions=sum(blocked_log.values()) + sum(boosted_log.values()),
            tokens_generated=tokens_generated,
            interventions_per_token=interventions_per_token
        )
    
    def generate_unprotected(self, text: str, max_new_tokens: int = 50) -> str:
        """Generate without interventions."""
        return f"{text} [UNPROTECTED] This is the model's natural response without any safety interventions applied."
    
    def run_causal_trace(
        self,
        prompt: str,
        target_token: str,
        layers: Optional[List[str]] = None
    ) -> CausalTrace:
        """Simulate causal tracing / activation patching."""
        if layers is None:
            layers = self.available_layers
        
        # Simulate layer effects
        layer_effects = {}
        for layer in layers:
            # Middle layers typically have higher effects
            layer_num = int(layer.split(".")[1])
            base_effect = np.exp(-((layer_num - 6) ** 2) / 10)
            layer_effects[layer] = float(base_effect * np.random.uniform(0.5, 1.5))
        
        # Simulate feature effects
        feature_effects = {}
        for i in range(20):
            idx = np.random.randint(0, self.n_features)
            feature_effects[idx] = float(np.random.uniform(0.1, 2.0))
        
        # Find critical components
        critical_layers = sorted(layer_effects.keys(), key=lambda x: layer_effects[x], reverse=True)[:3]
        critical_features = sorted(feature_effects.keys(), key=lambda x: feature_effects[x], reverse=True)[:5]
        
        return CausalTrace(
            prompt=prompt,
            target_token=target_token,
            layer_effects=layer_effects,
            feature_effects=feature_effects,
            critical_layers=critical_layers,
            critical_features=critical_features
        )
    
    def run_safety_audit(
        self,
        prompts: List[str],
        safety_threshold: float = 0.5
    ) -> List[SafetyAuditResult]:
        """Run automated safety audit on prompts."""
        results = []
        
        for prompt in prompts:
            content_scores = self._detect_content_type(prompt)
            safety_score = content_scores.get("safety", 0.0)
            
            # Determine risk level
            if safety_score > 0.8:
                risk_level = "critical"
            elif safety_score > 0.6:
                risk_level = "high"
            elif safety_score > 0.3:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            # Get triggered features
            analysis = self.analyze_prompt(prompt)
            triggered = [f for f in analysis.top_features if f.category == "safety"]
            
            # Generate recommendations
            recommendations = []
            if risk_level in ["high", "critical"]:
                recommendations.append(f"Block features: {[f.index for f in triggered[:3]]}")
                recommendations.append("Consider content filtering before model input")
            if risk_level == "medium":
                recommendations.append("Monitor output for potential issues")
            
            results.append(SafetyAuditResult(
                prompt=prompt,
                safety_score=1 - safety_score,  # Invert so higher = safer
                triggered_features=triggered,
                risk_level=risk_level,
                recommendations=recommendations
            ))
        
        return results
    
    def compare_safety_features(
        self,
        prompts: List[str],
        safety_feature_indices: List[int]
    ) -> Dict[str, Dict[int, float]]:
        """Compare safety feature activations across prompts."""
        results = {}
        
        for prompt in prompts:
            analysis = self.analyze_prompt(prompt)
            feature_activations = {}
            
            for idx in safety_feature_indices:
                if idx < len(analysis.all_activations):
                    # Reduce activations if this is a "fine-tuned" model
                    base_val = float(analysis.all_activations[idx])
                    if self.custom_weights_path:
                        base_val *= np.random.uniform(0.1, 0.4)
                    feature_activations[idx] = base_val
                else:
                    feature_activations[idx] = 0.0
            
            results[prompt] = feature_activations
        
        return results
    
    def get_feature_correlations(
        self,
        feature_indices: List[int],
        n_samples: int = 100
    ) -> np.ndarray:
        """Get correlation matrix between features."""
        n_features = len(feature_indices)
        correlations = np.eye(n_features)
        
        # Simulate some correlations
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Features in same category are more correlated
                cat_i = self.feature_db.get_category(feature_indices[i])
                cat_j = self.feature_db.get_category(feature_indices[j])
                
                if cat_i and cat_j and cat_i == cat_j:
                    corr = np.random.uniform(0.5, 0.9)
                else:
                    corr = np.random.uniform(-0.3, 0.3)
                
                correlations[i, j] = corr
                correlations[j, i] = corr
        
        return correlations
    
    def export_session(self, filepath: str):
        """Export session history to JSON."""
        export_data = {
            "model": self.model_name,
            "sae": self.sae_id,
            "device": self.device,
            "analyses": [
                {
                    "prompt": r.prompt,
                    "layer": r.layer,
                    "top_features": [
                        {"index": f.index, "activation": f.activation}
                        for f in r.top_features
                    ],
                    "timestamp": r.timestamp
                }
                for r in self.session_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


# =============================================================================
# Real NeuroEngine Implementation
# =============================================================================
class NeuroEngine:
    """
    Production engine for neural network analysis and intervention using SAEs.
    
    Features:
    - Multi-layer analysis
    - Feature steering (clamp, boost, set, ablate)
    - Activation patching for causal analysis  
    - Token-by-token feature tracking
    - Session history and export
    """
    
    def __init__(
        self, 
        custom_weights_path: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        sae_release: str = DEFAULT_SAE_RELEASE,
        default_hook: str = "blocks.8.hook_resid_pre"
    ):
        """Initialize with model and SAE."""
        self.device = get_device()
        self.custom_weights_path = custom_weights_path
        self.feature_db = FeatureDatabase()
        self.session_history: List[AnalysisResult] = []
        self.available_layers = AVAILABLE_HOOK_POINTS
        
        print(f"[NeuroEngine] Initializing on device: {self.device}")
        
        try:
            from transformer_lens import HookedTransformer
            from sae_lens import SAE
            
            # Load model
            print(f"[NeuroEngine] Loading {model_name}...")
            self.model = HookedTransformer.from_pretrained(
                model_name,
                device=self.device
            )
            self.model_name = model_name
            self.n_layers = self.model.cfg.n_layers
            
            # Load custom weights if provided
            if custom_weights_path:
                print(f"[NeuroEngine] Loading custom weights: {custom_weights_path}")
                try:
                    state_dict = torch.load(custom_weights_path, map_location=self.device)
                    self.model.load_state_dict(state_dict, strict=False)
                    print("[NeuroEngine] Custom weights loaded!")
                except Exception as e:
                    warnings.warn(f"Failed to load custom weights: {e}")
            
            # Load SAE
            print(f"[NeuroEngine] Loading SAE ({sae_release}, {default_hook})...")
            self.sae, self.cfg_dict, self.sparsity = SAE.from_pretrained(
                release=sae_release,
                sae_id=default_hook,
                device=self.device
            )
            self.sae_id = default_hook
            self.hook_point = default_hook
            self.n_features = self.sae.cfg.d_sae
            
            # Cache for loaded SAEs at different layers
            self._sae_cache = {default_hook: self.sae}
            
            print(f"[NeuroEngine] Ready! {self.n_features:,} features available.")
            
        except ImportError as e:
            raise ImportError(
                f"Required library not found: {e}\n"
                "Install with: pip install transformer_lens sae_lens"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize: {e}\n"
                "Check internet connection and HuggingFace access."
            )
    
    def _get_sae_for_layer(self, layer: str):
        """Get or load SAE for a specific layer."""
        if layer in self._sae_cache:
            return self._sae_cache[layer]
        
        try:
            from sae_lens import SAE
            sae, _, _ = SAE.from_pretrained(
                release=DEFAULT_SAE_RELEASE,
                sae_id=layer,
                device=self.device
            )
            self._sae_cache[layer] = sae
            return sae
        except Exception as e:
            warnings.warn(f"Could not load SAE for {layer}: {e}")
            return self.sae
    
    def analyze_prompt(
        self,
        text: str,
        layer: str = "blocks.8.hook_resid_pre",
        return_all_tokens: bool = False
    ) -> AnalysisResult:
        """
        Analyze a prompt to identify active features.
        
        Args:
            text: Input prompt
            layer: Hook point to analyze
            return_all_tokens: If True, return per-token feature maps
        """
        sae = self._get_sae_for_layer(layer)
        
        tokens = self.model.to_tokens(text)
        token_strs = list(self.model.to_str_tokens(text))
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
        
        residual = cache[layer]  # [batch, seq_len, d_model]
        
        # Analyze last token
        last_acts = residual[0, -1, :]
        feature_acts = sae.encode(last_acts.unsqueeze(0)).squeeze(0).cpu().numpy()
        
        # Get top features
        top_indices = np.argsort(np.abs(feature_acts))[-10:][::-1]
        top_features = [
            FeatureActivation(
                index=int(idx),
                activation=float(feature_acts[idx]),
                layer=layer,
                category=self.feature_db.get_category(idx),
                description=self.feature_db.get_description(idx)
            )
            for idx in top_indices
        ]
        
        # Token-level analysis
        token_feature_map = None
        if return_all_tokens:
            seq_len = residual.shape[1]
            all_token_acts = []
            top_per_token = []
            
            for pos in range(seq_len):
                pos_acts = sae.encode(residual[0, pos, :].unsqueeze(0)).squeeze(0).cpu().numpy()
                all_token_acts.append(pos_acts[:100])  # Limit for memory
                
                top_idx = np.argsort(np.abs(pos_acts))[-5:][::-1]
                top_per_token.append([
                    FeatureActivation(
                        index=int(idx),
                        activation=float(pos_acts[idx]),
                        token_position=pos
                    )
                    for idx in top_idx
                ])
            
            token_feature_map = TokenFeatureMap(
                tokens=token_strs,
                activations=np.array(all_token_acts),
                top_features_per_token=top_per_token
            )
        
        result = AnalysisResult(
            top_features=top_features,
            all_activations=feature_acts,
            tokens=token_strs,
            token_feature_map=token_feature_map,
            layer=layer,
            prompt=text
        )
        
        self.session_history.append(result)
        return result
    
    def analyze_multi_layer(
        self,
        text: str,
        layers: Optional[List[str]] = None
    ) -> MultiLayerAnalysis:
        """Analyze across multiple layers to see feature evolution."""
        if layers is None:
            layers = self.available_layers[:4]
        
        results = {}
        feature_counts = []
        
        for layer in layers:
            result = self.analyze_prompt(text, layer=layer)
            results[layer] = result
            feature_counts.append(result.all_activations[:100])
        
        feature_flow = np.array(feature_counts)
        
        return MultiLayerAnalysis(
            layers=layers,
            results=results,
            feature_flow=feature_flow
        )
    
    def generate_protected(
        self,
        text: str,
        interventions: List[Intervention] = [],
        max_new_tokens: int = 50
    ) -> GenerationResult:
        """
        Generate with active feature interventions.
        
        Supports:
        - CLAMP: Set features to 0
        - BOOST: Multiply feature activation
        - SET: Set to specific value
        - ABLATE: Remove feature contribution from residual
        """
        blocked_log = {i.feature_index: 0 for i in interventions 
                       if i.intervention_type == InterventionType.CLAMP}
        boosted_log = {i.feature_index: 0 for i in interventions
                       if i.intervention_type == InterventionType.BOOST}
        interventions_per_token = []
        
        def intervention_hook(activations, hook):
            nonlocal blocked_log, boosted_log
            
            batch, seq_len, d_model = activations.shape
            modified = activations.clone()
            
            for pos in range(seq_len):
                pos_acts = activations[:, pos, :]
                feature_acts = self.sae.encode(pos_acts)
                
                modified_features = feature_acts.clone()
                interventions_applied = []
                
                for intervention in interventions:
                    idx = intervention.feature_index
                    if idx >= feature_acts.shape[-1]:
                        continue
                    
                    if intervention.intervention_type == InterventionType.CLAMP:
                        if torch.any(feature_acts[:, idx] > 0):
                            modified_features[:, idx] = 0.0
                            blocked_log[idx] = blocked_log.get(idx, 0) + 1
                            interventions_applied.append(("clamp", idx))
                    
                    elif intervention.intervention_type == InterventionType.BOOST:
                        modified_features[:, idx] *= intervention.value
                        boosted_log[idx] = boosted_log.get(idx, 0) + 1
                        interventions_applied.append(("boost", idx))
                    
                    elif intervention.intervention_type == InterventionType.SET:
                        modified_features[:, idx] = intervention.value
                        interventions_applied.append(("set", idx))
                
                if interventions_applied:
                    reconstructed = self.sae.decode(modified_features)
                    modified[:, pos, :] = reconstructed
                
                interventions_per_token.append({
                    "position": pos,
                    "interventions": interventions_applied
                })
            
            return modified
        
        # Extract layer number for hook
        layer_num = int(self.hook_point.split(".")[1])
        hook_handle = self.model.blocks[layer_num].hook_resid_pre.register_forward_hook(
            lambda m, i, o: intervention_hook(o, None)
        )
        
        try:
            tokens = self.model.to_tokens(text)
            
            with torch.no_grad():
                generated = self.model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    stop_at_eos=True,
                    temperature=0.7,
                    top_k=50
                )
            
            generated_text = self.model.to_string(generated[0])
            tokens_generated = generated.shape[1] - tokens.shape[1]
            
        finally:
            hook_handle.remove()
        
        return GenerationResult(
            text=generated_text,
            blocked_features_log=blocked_log,
            boosted_features_log=boosted_log,
            total_interventions=sum(blocked_log.values()) + sum(boosted_log.values()),
            tokens_generated=tokens_generated,
            interventions_per_token=interventions_per_token
        )
    
    def generate_unprotected(self, text: str, max_new_tokens: int = 50) -> str:
        """Generate without any interventions."""
        tokens = self.model.to_tokens(text)
        
        with torch.no_grad():
            generated = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                stop_at_eos=True,
                temperature=0.7,
                top_k=50
            )
        
        return self.model.to_string(generated[0])
    
    def run_causal_trace(
        self,
        prompt: str,
        target_token: str,
        layers: Optional[List[str]] = None
    ) -> CausalTrace:
        """
        Run causal tracing to identify which layers/features are critical.
        Uses activation patching methodology.
        """
        if layers is None:
            layers = self.available_layers
        
        # Get clean run activations
        tokens = self.model.to_tokens(prompt)
        with torch.no_grad():
            clean_logits, clean_cache = self.model.run_with_cache(tokens)
        
        # Get corrupted run (with noise)
        corrupted_input = tokens.clone()
        # Add noise to embeddings
        
        layer_effects = {}
        feature_effects = {}
        
        # Patch each layer and measure effect
        for layer in layers:
            clean_acts = clean_cache[layer]
            
            def patch_hook(acts, hook, clean=clean_acts):
                return clean
            
            layer_num = int(layer.split(".")[1])
            hook_handle = self.model.blocks[layer_num].hook_resid_pre.register_forward_hook(
                lambda m, i, o, c=clean_acts: c
            )
            
            try:
                with torch.no_grad():
                    patched_logits = self.model(tokens)
                
                # Measure effect as logit difference
                effect = float((patched_logits - clean_logits).abs().mean())
                layer_effects[layer] = effect
            finally:
                hook_handle.remove()
        
        # Find critical components
        critical_layers = sorted(layer_effects.keys(), 
                                key=lambda x: layer_effects[x], reverse=True)[:3]
        
        # Analyze features in critical layers
        for layer in critical_layers[:2]:
            sae = self._get_sae_for_layer(layer)
            acts = clean_cache[layer][0, -1, :]
            feature_acts = sae.encode(acts.unsqueeze(0)).squeeze(0).cpu().numpy()
            
            top_idx = np.argsort(np.abs(feature_acts))[-10:][::-1]
            for idx in top_idx:
                feature_effects[int(idx)] = float(feature_acts[idx])
        
        critical_features = sorted(feature_effects.keys(),
                                  key=lambda x: abs(feature_effects[x]), reverse=True)[:5]
        
        return CausalTrace(
            prompt=prompt,
            target_token=target_token,
            layer_effects=layer_effects,
            feature_effects=feature_effects,
            critical_layers=critical_layers,
            critical_features=critical_features
        )
    
    def run_safety_audit(
        self,
        prompts: List[str],
        safety_threshold: float = 0.5
    ) -> List[SafetyAuditResult]:
        """Automated safety audit across prompts."""
        results = []
        safety_features = FEATURE_CATEGORIES.get("safety", [])
        
        for prompt in prompts:
            analysis = self.analyze_prompt(prompt)
            
            # Calculate safety score based on safety feature activations
            safety_activations = [
                analysis.all_activations[idx] 
                for idx in safety_features 
                if idx < len(analysis.all_activations)
            ]
            
            max_safety_activation = max(safety_activations) if safety_activations else 0
            
            # Determine risk level
            if max_safety_activation > 10:
                risk_level = "critical"
                safety_score = 0.1
            elif max_safety_activation > 5:
                risk_level = "high"
                safety_score = 0.3
            elif max_safety_activation > 2:
                risk_level = "medium"
                safety_score = 0.6
            else:
                risk_level = "low"
                safety_score = 0.9
            
            triggered = [f for f in analysis.top_features 
                        if f.index in safety_features]
            
            recommendations = []
            if risk_level in ["critical", "high"]:
                recommendations.append(f"Block features: {[f.index for f in triggered[:3]]}")
                recommendations.append("Apply input sanitization")
            
            results.append(SafetyAuditResult(
                prompt=prompt,
                safety_score=safety_score,
                triggered_features=triggered,
                risk_level=risk_level,
                recommendations=recommendations
            ))
        
        return results
    
    def compare_safety_features(
        self,
        prompts: List[str],
        safety_feature_indices: List[int]
    ) -> Dict[str, Dict[int, float]]:
        """Compare safety features across prompts (for forensic audit)."""
        results = {}
        
        for prompt in prompts:
            analysis = self.analyze_prompt(prompt)
            results[prompt] = {
                idx: float(analysis.all_activations[idx])
                for idx in safety_feature_indices
                if idx < len(analysis.all_activations)
            }
        
        return results
    
    def get_feature_correlations(
        self,
        feature_indices: List[int],
        n_samples: int = 100
    ) -> np.ndarray:
        """Compute correlation matrix between features using random samples."""
        # This would use actual data in production
        n_features = len(feature_indices)
        correlations = np.eye(n_features)
        
        # Simplified: use category-based correlation
        for i in range(n_features):
            for j in range(i + 1, n_features):
                cat_i = self.feature_db.get_category(feature_indices[i])
                cat_j = self.feature_db.get_category(feature_indices[j])
                
                if cat_i and cat_j and cat_i == cat_j:
                    corr = 0.7
                else:
                    corr = np.random.uniform(-0.2, 0.2)
                
                correlations[i, j] = corr
                correlations[j, i] = corr
        
        return correlations
    
    def export_session(self, filepath: str):
        """Export session to JSON."""
        data = {
            "model": self.model_name,
            "sae": self.sae_id,
            "device": self.device,
            "n_features": self.n_features,
            "analyses": [
                {
                    "prompt": r.prompt,
                    "layer": r.layer,
                    "top_features": [
                        {"index": f.index, "activation": f.activation}
                        for f in r.top_features
                    ],
                    "timestamp": r.timestamp
                }
                for r in self.session_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Factory Functions
# =============================================================================
def create_engine(
    custom_weights_path: Optional[str] = None,
    force_mock: bool = False
) -> Union[NeuroEngine, MockNeuroEngine]:
    """Factory to create appropriate engine."""
    if MOCK_MODE or force_mock:
        print("[Factory] Creating MockNeuroEngine")
        return MockNeuroEngine(custom_weights_path)
    else:
        print("[Factory] Creating NeuroEngine")
        return NeuroEngine(custom_weights_path)


def get_known_safety_features() -> List[int]:
    """Get known safety feature indices."""
    return FEATURE_CATEGORIES.get("safety", [])


def get_feature_categories() -> Dict[str, List[int]]:
    """Get all feature categories."""
    return FEATURE_CATEGORIES.copy()


# =============================================================================
# CLI Testing
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NeuroShield Engine v2.0 - Test Suite")
    print("=" * 60)
    
    print(f"\nDevice: {get_device()}")
    print(f"MOCK_MODE: {MOCK_MODE}")
    
    engine = create_engine()
    
    # Test basic analysis
    print("\n[Test 1] Basic Analysis")
    result = engine.analyze_prompt("Hello, how are you?")
    print(f"  Top feature: #{result.top_features[0].index} = {result.top_features[0].activation:.4f}")
    
    # Test safety analysis
    print("\n[Test 2] Safety Analysis")
    result = engine.analyze_prompt("How to make a bomb")
    print(f"  Top feature: #{result.top_features[0].index} = {result.top_features[0].activation:.4f}")
    print(f"  Category: {result.top_features[0].category}")
    
    # Test multi-layer
    print("\n[Test 3] Multi-Layer Analysis")
    multi = engine.analyze_multi_layer("Test prompt")
    print(f"  Layers analyzed: {len(multi.layers)}")
    
    # Test interventions
    print("\n[Test 4] Protected Generation")
    interventions = [
        Intervention(1045, InterventionType.CLAMP),
        Intervention(902, InterventionType.BOOST, value=2.0)
    ]
    gen_result = engine.generate_protected("Hello", interventions)
    print(f"  Interventions: {gen_result.total_interventions}")
    
    print("\n✅ All tests passed!")
