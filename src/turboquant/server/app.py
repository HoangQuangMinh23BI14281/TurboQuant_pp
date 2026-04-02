import os
import time
import json
import uuid
import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import AsyncGenerator, Dict, Optional

from turboquant.integrations.patcher import patch_hf_model
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.cache.block_pool import KVBlockPool
from turboquant.server.models import (
    ChatCompletionRequest, ChatCompletionResponse, 
    ChatCompletionResponseChoice, ChatCompletionUsage,
    ChatMessage, ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice
)

app = FastAPI(title="TurboQuant++ SOTA Server")

# Global State
model = None
tokenizer = None
pool = None
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

@app.on_event("startup")
def startup_event():
    global model, tokenizer, pool
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[*] Loading Model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    
    print("[*] Applying Surgical TurboQuant Patcher...")
    model = patch_hf_model(model)
    
    print("[*] Initializing 2GB Global KVBlockPool...")
    # 1024 blocks of 128 each = 128k total tokens (approx 256MB at 4-bit)
    n_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    pool = KVBlockPool(num_blocks=1024, head_dim=head_dim, n_heads=n_heads, device=device)
    print(f"[*] Server Ready on {device}")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Format Prompt
    # vLLM-style simple chat template concatenation
    full_prompt = ""
    for msg in request.messages:
        full_prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    full_prompt += "<|im_start|>assistant\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    
    # 2. Allocate SOTA Request Cache
    request_id = f"chatcmpl-{uuid.uuid4()}"
    
    # SOTA: Configure GQA properly for the cache
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    group_size = n_heads // n_kv_heads

    cache = TurboQuantKVCache(layer_idx=1, pool=pool)
    cache.n_heads = n_heads
    cache.n_kv_heads = n_kv_heads
    cache.group_size = group_size
    
    if request.stream:
        return StreamingResponse(
            stream_generator(request_id, inputs, cache, request),
            media_type="text/event-stream"
        )
    
    # 3. Non-Streaming Path
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=request.max_tokens, 
            temperature=request.temperature,
            past_key_values=cache,
            use_cache=True
        )
        
    completion_text = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
    completion_len = output.shape[1] - prompt_len
    
    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=completion_text),
                finish_reason="stop"
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_len,
            completion_tokens=completion_len,
            total_tokens=prompt_len + completion_len
        )
    )

async def stream_generator(request_id: str, inputs: Dict, cache: TurboQuantKVCache, request: ChatCompletionRequest):
    """SOTA SSE Stream Generator"""
    # Simply wrapping model.generate() for streaming in HF is complex, 
    # but for demo purposes we use a simple loop.
    
    curr_input_ids = inputs.input_ids
    generated_tokens = 0
    
    # Pre-fill
    with torch.no_grad():
        logits = model(curr_input_ids, past_key_values=cache, use_cache=True).logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        
    # Yield first chunk
    chunk_text = tokenizer.decode(next_token[0])
    yield f"data: {json.dumps(create_chunk(request_id, request.model, chunk_text))}\n\n"
    
    # Decode Loop
    for _ in range(request.max_tokens - 1):
        with torch.no_grad():
            logits = model(next_token, past_key_values=cache, use_cache=True).logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
        token_id = next_token.item()
        if token_id == tokenizer.eos_token_id:
            break
            
        chunk_text = tokenizer.decode(next_token[0])
        yield f"data: {json.dumps(create_chunk(request_id, request.model, chunk_text))}\n\n"
        generated_tokens += 1
        
    yield "data: [DONE]\n\n"

def create_chunk(id: str, model: str, text: str):
    return ChatCompletionStreamResponse(
        id=id,
        created=int(time.time()),
        model=model,
        choices=[ChatCompletionStreamResponseChoice(
            index=0,
            delta={"content": text},
            finish_reason=None
        )]
    ).dict()
