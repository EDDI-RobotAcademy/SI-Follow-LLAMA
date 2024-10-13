from dataset.service.dataset_service import DatasetService
from dataset.repository.dataset_repository_impl import DatasetRepositoryImpl


class DatasetServiceImpl(DatasetService):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__dataset_repository = DatasetRepositoryImpl.get_instance()

        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    def tokenize_dataset(self, dataset_id, tokenizer):
        dataset = self.__dataset_repository.tokenize_dataset(dataset_id, tokenizer)
        return dataset
