
import requests

from config import (
    OLLAMA_BASE_URL
)

MODEL_NAME_MAP = {
    "Qwen/Qwen3.5-9B-Instruct":  "qwen3.5:latest",
    "Gemma/Gemma3-1B": "gemma3:1b",
    "Qwen/Qwen2-math-1.5B": "qwen2-math:1.5b",
    "Llama/Llama3.2-1.5B": "llama3.2:1b"
}

def load_model(model_name: str):
    """
        This just checks whether the ollama models are properly running and are avaialbe or not

        Returns the model_name, None
    """
    ollamaName = MODEL_NAME_MAP.get(model_name, model_name)

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
    if not any(ollamaName in a for a in available):
        print(f" ERROR: Ollama model is not avialbe. Please check again after properly downloing the model")
        print(f"  Run: ollama pull {ollamaName} and check for the models")

    print(f"Ollama ready. Using model: {ollamaName}")
    return ollamaName, None  # (model, tokenizer) — tokenizer is None with Ollama

def generate_response(model, promptText, maxTokens: int = 8192, temperature: float = 0) -> str:
    """
    Generate a response using Ollma model

    Returns:
        str: Get the response from the ollama model.
    """

    ollamaName = MODEL_NAME_MAP.get(model, model)

    if "qwen" in ollamaName.lower():
        promptText = promptText + "\n/no_think"

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": ollamaName,
                "messages": [{"role": "user", "content": promptText}],
                "think": False,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": maxTokens
                }
            },
            timeout=300
        )
        response.raise_for_status()
        result = response.json().get("message", {}).get("content", "").strip()
        print(f"Response: {result}")
        return result
    except requests.exceptions.Timeout:
        print(f"  WARNING: Request timed out for model {ollamaName}")
        return ""
    except Exception as e:
        print(f"  ERROR: Ollama request failed: {e}")
        return ""

def free_model(model):
    """
    Nothing to free with Ollama — the server manages memory.
    """
    print(f"  Model '{model}' released (Ollama manages memory automatically).")
    print(f"  You may need to stop the ollama model manually if the memory is not freed. RUN: ollama stop {model}")
