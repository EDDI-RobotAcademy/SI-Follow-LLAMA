from abc import ABC, abstractmethod


class DatasetService(ABC):
    @abstractmethod
    def tokenize_dataset(self, dataset_id, tokenizer):
        pass
    
    @abstractmethod
    def get_data_collator(self, tokenizer):
        pass