from abc import ABC, abstractmethod


class DatasetRepository(ABC):    
    @abstractmethod
    def load_dataset(self, dataset_id, tokenizer):
        pass
    
