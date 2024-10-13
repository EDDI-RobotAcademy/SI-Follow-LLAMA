import subprocess
from glob import glob

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from to_ollama.repository.to_ollama_repository import ToOllamaRepository


class ToOllamaRepositoryImpl(ToOllamaRepository):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        return cls.__instance

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    def merge_adapter(self, base_model_id, adapter_model_id, save_path):
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            return_dict=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        model = PeftModel.from_pretrained(
            base_model, adapter_model_id, device_map="cuda"
        )
        model = model.merge_and_unload()

        model.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)

    def to_gguf(self, model_path):
        subprocess.Popen(
            f"python to_ollama/entity/llama_cpp/convert_hf_to_gguf.py {model_path}",
            shell=True,
        ).wait()

