from dataclasses import dataclass, field
from typing import Optional, Dict
from turboquant.cache.routing import QuantizationStrategy

@dataclass
class TurboQuantConfig:
    """
    Configuration for TurboQuant++ Hybrid Precision & Boundary Protection.
    
    Attributes:
        k_bits: Precision for Key cache (default: 8-bit for high accuracy).
        v_bits: Precision for Value cache (default: 3-bit or 4-bit for Maxwell/Pascal/Hopper).
        protect_boundaries: If True, bypass quantization for first/last N/M layers.
        n_head_protected: Number of initial layers to protect.
        n_tail_protected: Number of final layers to protect.
        group_size: Quantization group size (default: 128).
    """
    k_bits: int = 5    # SOTA Default: 4-bit MSE + 1-bit Sign
    v_bits: int = 8    # SOTA Default: High-fidelity Value
    protect_boundaries: bool = True
    n_head_protected: int = 2
    n_tail_protected: int = 2
    group_size: int = 128
    v_group_size: int = 32
    n_rotation_passes: int = 1
    qjl_scale: float = 0.1 # Calibrated SOTA Scale
    quest_threshold: float = -1e6 # Default: Disable sparsity for stability
    
    # Advanced routing (Layer-specific overrides)
    layer_overrides: Dict[int, Dict] = field(default_factory=dict)

    def is_protected(self, layer_idx: int, total_layers: int) -> bool:
        """Determines if a specific layer should remain in FP16."""
        if not self.protect_boundaries:
            return False
        
        # Check explicit layer overrides
        if layer_idx in self.layer_overrides:
            return self.layer_overrides[layer_idx].get("protected", False)
            
        # Standard head/tail protection
        is_head = layer_idx < self.n_head_protected
        is_tail = layer_idx >= (total_layers - self.n_tail_protected)
        
        return is_head or is_tail

    def get_bits(self, layer_idx: int) -> tuple:
        """Returns (k_bits, v_bits) for a specific layer."""
        if layer_idx in self.layer_overrides:
            override = self.layer_overrides[layer_idx]
            return override.get("k_bits", self.k_bits), override.get("v_bits", self.v_bits)
        return self.k_bits, self.v_bits

    def get_strategy(self, layer_idx: int, total_layers: int) -> QuantizationStrategy:
        """Determines the quantization strategy for a specific layer."""
        if self.is_protected(layer_idx, total_layers):
            return QuantizationStrategy.FP16
            
        # Standard: use what's configured (defaulting to 4bit which is our standard hardened path)
        return QuantizationStrategy.TURBO_4BIT
