import torch
import torch.nn as nn
from typing import Optional
from .wht import fwht, ifwht
from .sign_array import generate_sign_array, apply_sign_array

class TurboQuantRotation(nn.Module):
    def __init__(self, d: int, n_passes: int = 2, pattern: str = 'tbq'):
        super().__init__()
        self.d = d
        self.n_passes = n_passes
        self.pattern = pattern
        all_signs = []
        for i in range(n_passes):
            p = pattern if i == 0 else f'{pattern}_{i}'
            signs = generate_sign_array(d, use_llama_preset=p if i == 0 else None)
            all_signs.append(signs)
        self.register_buffer('all_signs', torch.stack(all_signs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i in range(self.n_passes):
            out = apply_sign_array(out, self.all_signs[i])
            out = fwht(out)
        return out

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i in reversed(range(self.n_passes)):
            out = ifwht(out)
            out = apply_sign_array(out, self.all_signs[i])
        return out

def apply_cascaded_srht(x: torch.Tensor, n_passes: int = 2, pattern: str = 'tbq') -> torch.Tensor:
    rot = TurboQuantRotation(x.shape[-1], n_passes=n_passes, pattern=pattern).to(x.device)
    return rot(x)
