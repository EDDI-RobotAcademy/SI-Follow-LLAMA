from train.repository.train_repository_impl import TrainRepositoryImpl
from train.service.train_service import TrainService
class TrainServiceImpl(TrainService):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__train_repository = TrainRepositoryImpl.get_instance()

        return cls.__instance

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance
