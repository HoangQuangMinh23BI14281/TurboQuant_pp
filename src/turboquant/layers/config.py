from dataclasses import dataclass, field
from typing import Optional, Dict
from turboquant.cache.routing import QuantizationStrategy

@dataclass
class QuantConfig:
    """
    Quantization-specific parameters.
    """
    k_bits: int = 5    # SOTA Default: 4-bit MSE + 1-bit Sign
    v_bits: int = 3   # SOTA Default: High-fidelity Value
    k_group_size: int = 64
    v_group_size: int = 32
    n_rotation_passes: int = 1
    qjl_scale: float = 0.1 # Calibrated SOTA Scale
    quant_epsilon: float = 1e-10 # Numerical stability constant
    v_scale_epsilon: float = 1e-6

@dataclass
class HardwareConfig:
    """
    Execution and memory-specific parameters.
    """
    num_blocks: int = 1024
    tokens_per_block: int = 128
    hardware_alignment: int = 128
    triton_block_n: int = 128
    triton_num_warps: int = 4

@dataclass
class TurboQuantConfig:
    """
    Configuration for TurboQuant++ Hybrid Precision & Boundary Protection.
    """
    quant: QuantConfig = field(default_factory=QuantConfig)
    hw: HardwareConfig = field(default_factory=HardwareConfig)
    
    # Boundary Protection (Routing)
    protect_boundaries: bool = True
    n_head_protected: int = 2
    n_tail_protected: int = 2
    max_seq_len: int = 4096
    rope_base: int = 1_000_000   # Qwen2.5 Base (Standard SOTA)
    
    # Advanced routing (Layer-specific overrides)
    sm_scale: Optional[float] = None
    quest_threshold: float = -1e6 # Default: Disable sparsity for stability
    layer_overrides: Dict[int, Dict] = field(default_factory=dict)

    def __post_init__(self):
        # SOTA Pillar 1: Ensure QuantConfig and HardwareConfig are objects
        if not isinstance(self.quant, QuantConfig):
            self.quant = QuantConfig(**(self.quant if isinstance(self.quant, dict) else {}))
        if not isinstance(self.hw, HardwareConfig):
            self.hw = HardwareConfig(**(self.hw if isinstance(self.hw, dict) else {}))

        # SOTA Pillar 3: Hardware Invariant (Triton Tile <= Paged Block)
        # Nếu triton_block_n > tokens_per_block, kernel sẽ đọc tràn sang block vật lý khác.
        if self.hw.triton_block_n > self.hw.tokens_per_block:
            self.hw.triton_block_n = self.hw.tokens_per_block

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
            return override.get("k_bits", self.quant.k_bits), override.get("v_bits", self.quant.v_bits)
        return self.quant.k_bits, self.quant.v_bits

    def get_strategy(self, layer_idx: int, total_layers: int) -> QuantizationStrategy:
        """Determines the quantization strategy for a specific layer."""
        if self.is_protected(layer_idx, total_layers):
            return QuantizationStrategy.FP16
            
        # Standard: use what's configured (defaulting to 4bit which is our standard hardened path)
        return QuantizationStrategy.TURBO_4BIT
