import uvicorn
import argparse
import os

def run():
    parser = argparse.ArgumentParser(description="TurboQuant++ SOTA Server (OpenAI Compatible)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace Model Path")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--device", type=str, default="cuda", help="Generation device")
    
    args = parser.parse_args()
    
    # Set environment variables for app.py to consume
    os.environ["MODEL_NAME"] = args.model
    os.environ["TQ_DEVICE"] = args.device
    
    print(f"\n🚀 Launching TurboQuant++ SOTA Server")
    print(f"📍 Model: {args.model}")
    print(f"📍 URL: http://{args.host}:{args.port}/v1")
    
    uvicorn.run("turboquant.server.app:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    run()
