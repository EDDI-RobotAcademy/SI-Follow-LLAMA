from to_ollama.repository.to_ollama_repository_impl import ToOllamaRepositoryImpl
from to_ollama.service.to_ollama_service import ToOllamaService


class ToOllamaServiceImpl(ToOllamaService):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__to_ollama_repository = ToOllamaRepositoryImpl.get_instance()

        return cls.__instance

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance
    
    def to_ollama(self, base_model_id, adapter_model_id, model_path):
        self.__to_ollama_repository.merge_adapter(base_model_id, adapter_model_id, model_path)
        self.__to_ollama_repository.to_gguf(model_path)
        self.__to_ollama_repository.make_modelfile(model_path)
        self.__to_ollama_repository.to_ollama(model_path)
