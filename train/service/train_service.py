from abc import ABC, abstractmethod


class TrainService(ABC):
    @abstractmethod
    def sft(self):
        pass
