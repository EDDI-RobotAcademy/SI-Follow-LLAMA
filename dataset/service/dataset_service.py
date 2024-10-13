from abc import ABC, abstractmethod


class DatasetService(ABC):
    @abstractmethod
    def tokenize_dataset(self, dataset_id, tokenizer):
        pass
    
