
import requests

from config import (
    OLLAMA_BASE_URL
)

MODEL_NAME_MAP = {
    "Qwen/Qwen3.5-9B-Instruct":  "qwen3.5:latest"
}

def load_model(model_name: str):
    """
        This just checks whether the ollama models are properly running and are avaialbe or not

        Returns the model_name, None
    """
    ollama_name = MODEL_NAME_MAP.get(model_name, model_name)

    # Checking if the ollama model is present on the server or not
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout = 5)
        # print(response)
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            f"Make sure you ran 'ollama serve' on the server to start ollama. Error: {e}"
        )
    
    # Now checking if the model is downloaded or not
    available = [m["name"] for m in response.json().get("models", [])]
    if not any(ollama_name in a for a in available):
        print(f" WARNING: Ollama model is not avialbe. Please check again after properly downloing the model")
        print(f"  Run: ollama pull {ollama_name} and check for the models")

    print(f"Ollama ready. Using model: {ollama_name}")
    return ollama_name, None  # (model, tokenizer) — tokenizer is None with Ollama


