import time
import requests
import runpod

import base64

# with open("assets/logo.png", "rb") as f:
#     img_bytes = f.read()

# data_url = "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")

VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"

def wait_for_vllm(timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get("http://127.0.0.1:8000/health", timeout=2)
            if r.status_code == 200:
                print("✅ VLLM server is ready")
                return True
        except Exception:
            time.sleep(2)
    raise RuntimeError("❌ VLLM server did not start in time")

# wait_for_vllm()

def handler(event):
    """
    event looks like:
    {
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "<base64-encoded image>"}},
                        {"type": "text", "text": "<|img|><|imgpad|><|endofimg|>What is in this image?"}
                    ]
                }
            ],
            "model": "model",
            "max_completion_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    """
    payload = event.get("input", {})
    # payload['messages'][0]['content'][0]['image_url']['url'] = data_url
    try:
        response = requests.post(
            VLLM_URL,
            json={
                "model": payload.get("model", "model"),  # must match served-model-name
                "messages": payload.get("messages", []),
                "max_tokens": payload.get("max_completion_tokens", 512),
                "temperature": payload.get("temperature", 0.7),
                "top_p": payload.get("top_p", 0.9),
            },
            timeout=240
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
