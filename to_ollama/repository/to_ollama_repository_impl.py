import os
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
            f"export PYTHONPATH={os.path.join(os.path.abspath(__file__), '..', '..', '..')};python -m to_ollama.entity.llama_cpp.convert_hf_to_gguf {model_path}",
            shell=True,
        ).wait()

    def make_modelfile(self, model_path):
        template = (
            f'FROM {glob(f"{model_path}/*.gguf")[0].split("/")[-1]}\n'
            'TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>\n'
            "{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>\n"
            "{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>\n"
            '{{ .Response }}<|eot_id|>"""\n'
            'PARAMETER stop "<|start_header_id|>"\n'
            'PARAMETER stop "<|end_header_id|>"\n'
            'PARAMETER stop "<|eot_id|>"\n'
            'PARAMETER stop "<|reserved_special_token"'
        )
        with open(f"{model_path}/Modelfile", "w") as f:
            f.write(template)

    def to_ollama(self, model_path):
        model_name = model_path.split('/')[-1]
        modelfile = glob(f"{model_path}/Modelfile")[0]
        subprocess.Popen(f"ollama create {model_name} -f {modelfile}", shell=True).wait()
