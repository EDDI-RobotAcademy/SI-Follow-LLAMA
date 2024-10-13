from abc import ABC, abstractmethod


class ToOllamaService(ABC):
    @abstractmethod
    def to_ollama(self, base_model_id, adapter_model_id, model_path):
        pass