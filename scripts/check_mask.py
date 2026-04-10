from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

print("hasattr _update_causal_mask:", hasattr(Qwen2Model, '_update_causal_mask'))
