import torch
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
from auto_gptq import AutoGPTQForCausalLM

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from constants import (
    CONTEXT_WINDOW_SIZE,
    MAX_NEW_TOKENS,
    N_GPU_LAYERS,
    N_BATCH,
    MODELS_PATH,
    INGEST_THREADS
)


def load_mps_model(model_id, model_basename, device_type, logging):
    try:
        logging.info("Using Llamacpp for  quantized models")
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=model_basename,
            resume_download=True,
            cache_dir=MODELS_PATH,
            
        )
        kwargs = {
            "model_path": model_path,
            "n_ctx": CONTEXT_WINDOW_SIZE,
            "max_tokens": MAX_NEW_TOKENS,
            "local_files_only": True,
            "n_batch": N_BATCH,  # set this based on your GPU & CPU RAM
            #"vocab_only": True,
            "use_mlock": False,
            "verbose": False,
        }
        if device_type.lower() == "mps":
            kwargs["n_gpu_layers"] = 0
        if device_type.lower() == "cuda":
            kwargs["n_gpu_layers"] = N_GPU_LAYERS  # set this based on your GPU

        return LlamaCpp(**kwargs)
    except:
        return None


def load_quantized_model_qptq(model_id, model_basename, device_type, logging):
    """
    Load a GPTQ quantized model using AutoGPTQForCausalLM.

    This function loads a quantized model that ends with GPTQ and may have variations
    of .no-act.order or .safetensors in their HuggingFace repo.

    Parameters:
    - model_id (str): The identifier for the model on HuggingFace Hub.
    - model_basename (str): The base name of the model file.
    - device_type (str): The type of device where the model will run.
    - logging (logging.Logger): Logger instance for logging messages.

    Returns:
    - model (AutoGPTQForCausalLM): The loaded quantized model.
    - tokenizer (AutoTokenizer): The tokenizer associated with the model.

    Notes:
    - The function checks for the ".safetensors" ending in the model_basename and removes it if present.
    """

    # The code supports all huggingface models that ends with GPTQ and have some variation
    # of .no-act.order or .safetensors in their HF repo.
    logging.info("Using AutoGPTQForCausalLM for quantized models")

    if ".safetensors" in model_basename:
        # Remove the ".safetensors" ending if present
        model_basename = model_basename.replace(".safetensors", "")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    logging.info("Tokenizer loaded")

    model = AutoGPTQForCausalLM.from_quantized(
        model_id,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device_map="auto",
        use_triton=False,
        quantize_config=None,
    )
    return model, tokenizer


def load_o_model(model_id, device_type, logging):
    if device_type.lower() in ["mps", "cpu"]:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(
            model_id, cache_dir="./models/")
        model = LlamaForCausalLM.from_pretrained(
            model_id, local_files_only=False, cache_dir="./models/")
    else:
        logging.info("Using AutoModelForCausalLM for other models")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir="./models/")
        logging.info("Tokenizer loaded")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=MODELS_PATH,
            # trust_remote_code=True, # set these if you are using NVIDIA GPU
            # load_in_4bit=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.float16,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    return model, tokenizer
