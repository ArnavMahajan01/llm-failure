import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME_MAP = {'Qwen/Qwen2.5-7B-Instruct': 'Qwen/Qwen2.5-7B-Instruct',
 'google/gemma-7b-it': 'google/gemma-7b-it'}
LOAD_IN_4BIT = True


def _compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _resolve_model_id(model_name: str) -> str:
    return MODEL_NAME_MAP.get(model_name, model_name)


def load_model(model_name: str):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is not available. In Colab, choose Runtime > Change runtime type > GPU.')

    model_id = _resolve_model_id(model_name)
    token = os.environ.get('HF_TOKEN') or None
    dtype = _compute_dtype()

    print(f'Loading HF model: {model_id}')
    print(f'Quantization: {"4-bit" if LOAD_IN_4BIT else "none"}; dtype: {dtype}')

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if LOAD_IN_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )

    model_kwargs = {
        'token': token,
        'device_map': 'auto',
        'torch_dtype': dtype,
        'trust_remote_code': True,
        'low_cpu_mem_usage': True,
    }
    if quantization_config is not None:
        model_kwargs['quantization_config'] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    return {'model': model, 'tokenizer': tokenizer, 'model_id': model_id}, tokenizer


def generate_response(model_bundle, promptText, maxTokens: int = 1024, temperature: float = 0) -> str:
    model = model_bundle['model']
    tokenizer = model_bundle['tokenizer']
    model_id = model_bundle['model_id']

    prompt = promptText
    if 'qwen3' in model_id.lower():
        prompt = prompt + '\n/no_think'

    if getattr(tokenizer, 'chat_template', None):
        text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors='pt')
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    do_sample = temperature is not None and temperature > 0

    generation_kwargs = {
        'max_new_tokens': maxTokens,
        'do_sample': do_sample,
        'pad_token_id': tokenizer.eos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs['temperature'] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = output_ids[0][inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print('Response:', response[:1000])
    return response


def free_model(model_bundle):
    model_id = model_bundle.get('model_id', 'unknown') if isinstance(model_bundle, dict) else 'unknown'
    if isinstance(model_bundle, dict):
        model_bundle.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Model '{model_id}' released from the notebook process.")
