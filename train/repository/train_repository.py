from abc import ABC, abstractmethod


class TrainRepository(ABC):
    @abstractmethod
    def load_model(self, model_id, model_config):
        pass

    @abstractmethod
    def load_tokenizer(self, model_id):
        pass


    @abstractmethod
    def get_lora_target(self, lora_target_config):
        pass

    @abstractmethod
    def get_lora_config(self, lora_config):
        pass

    @abstractmethod
    def get_device(self, **kwargs):
        pass

    @abstractmethod
    def get_train_args(self, train_args):
        pass

    @abstractmethod
    def get_trainer(
        self,
        model,
        tokenizer,
        train_args,
        train_dataset,
        eval_dataset,
        callbacks,
        data_collator,
    ):
        pass

    @abstractmethod
    def train(self, trainer):
        pass

    @abstractmethod
    def save_model(self, model, save_path):
        pass

