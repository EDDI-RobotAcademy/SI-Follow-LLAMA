from abc import ABC, abstractmethod


class ToOllamaRepository(ABC):
    @abstractmethod
    def merge_adapter(self, base_model_id, adapter_model_id):
        pass
    
    @abstractmethod
    def to_gguf(self, base_model_id, adapter_model_id):
        pass
    
