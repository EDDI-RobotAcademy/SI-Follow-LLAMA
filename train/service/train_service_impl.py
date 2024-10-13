from train.entity.configs.model_args import ModelCofig
from train.entity.configs.trainer_args import TrainerArgs
from train.repository.train_repository_impl import TrainRepositoryImpl
from train.service.train_service import TrainService
class TrainServiceImpl(TrainService):
    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
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
