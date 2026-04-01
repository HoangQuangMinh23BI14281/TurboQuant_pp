from enum import Enum, auto
from typing import Dict, List, Optional, Set

class QuantizationStrategy(Enum):
    FP16 = auto()       # No compression
    INT8 = auto()       # Fallback 8-bit
    TURBO_4BIT = auto() # SOTA 4-bit Dual-LUT (TurboQuant++)

class LayerRouting:
    """
    Manages the 'Boundary Protection' strategy for LLM layers.
    Determines which layers are exempt from 4-bit quantization to preserve
    semantic density at the early/late stages of the forward pass.
    """
    def __init__(
        self,
        num_layers: int,
        exempt_layers: Optional[List[int]] = None,
        strategy: QuantizationStrategy = QuantizationStrategy.TURBO_4BIT
    ):
        self.num_layers = num_layers
        self.strategy = strategy
        
        # Default: Exempt layer 0 and the final layer
        if exempt_layers is None:
            self.exempt_layers = {0, num_layers - 1}
        else:
            self.exempt_layers = set(exempt_layers)

    def get_strategy(self, layer_idx: int) -> QuantizationStrategy:
        """Route to appropriate strategy based on layer index."""
        if layer_idx in self.exempt_layers:
            return QuantizationStrategy.FP16
        return self.strategy

    @classmethod
    def from_percent(cls, num_layers: int, percent: float = 0.05):
        """Exempt first and last X% of layers."""
        n_exempt = max(1, int(num_layers * percent))
        exempt = list(range(n_exempt)) + list(range(num_layers - n_exempt, num_layers))
        return cls(num_layers, exempt_layers=exempt)
