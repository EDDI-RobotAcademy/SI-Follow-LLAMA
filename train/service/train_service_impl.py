from peft import get_peft_model

from dataset.repository.dataset_repository_impl import DatasetRepositoryImpl
from train.entity.configs.lora.lora_args import LoraArgs
from train.entity.configs.lora.lora_target import LoraTragets
from train.entity.configs.model_args import ModelCofig
from train.entity.configs.quantize_args import QuantizeArgs
from train.entity.configs.trainer_args import TrainerArgs
from train.repository.train_repository_impl import TrainRepositoryImpl
from train.service.train_service import TrainService


class TrainServiceImpl(TrainService):
    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
    DATASET_ID = "OneBottleKick/si-follow-dummy"
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__train_repository = TrainRepositoryImpl.get_instance()
            cls.__instance.__dataset_repository = DatasetRepositoryImpl.get_instance()

        return cls.__instance

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    def sft(self, save_path):
        quantize_config = self.__train_repository.get_quantization_config(
            QuantizeArgs()
        )

        model = self.__train_repository.load_model(
            self.MODEL_ID, ModelCofig(), quantize_config
        )
        tokenizer = self.__train_repository.load_tokenizer(self.MODEL_ID)

        lora_targets = self.__train_repository.get_lora_target(LoraTragets())
        lora_config = self.__train_repository.get_lora_config(LoraArgs(), lora_targets)

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        dataset = self.__dataset_repository.load_dataset(self.DATASET_ID, tokenizer)
        train_dataset = dataset
        eval_dataset = dataset
        # train_dataset = dataset
        # eval_dataset = None
        data_collator = self.__dataset_repository.get_data_collator(tokenizer)

        # TODO
        callbacks = None

        train_args = self.__train_repository.get_train_args(TrainerArgs())
        trainer = self.__train_repository.get_trainer(
            model,
            tokenizer,
            train_args,
            train_dataset,
            eval_dataset,
            callbacks,
            data_collator,
        )

        self.__train_repository.train(trainer)
        self.__train_repository.save_model(model, save_path)
