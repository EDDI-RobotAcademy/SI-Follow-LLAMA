from typing import List, Optional

import torch
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from train.entity.configs.quantize_args import QuantizeArgs
from train.repository.train_repository import TrainRepository


class TrainRepositoryImpl(TrainRepository):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        return cls.__instance

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    def load_model(
        self, model_id, model_config, quantize_config: Optional[QuantizeArgs] = None
    ):
        device = self.get_device()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=model_config.torch_dtype,
            low_cpu_mem_usage=True,
            use_cache=False,
            device_map=0,
            max_length=model_config.max_length,
            quantization_config=quantize_config,
        )
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.is_parallelizable = False
        model.model_parallel = False
        if quantize_config is not None:
            model = prepare_model_for_kbit_training(model)

        return model

    def load_tokenizer(self, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def get_quantization_config(self, quantize_config):
        quantize_config = BitsAndBytesConfig(
            load_in_4bit=quantize_config.load_in_4bit,
            bnb_4bit_quant_type=quantize_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=quantize_config.bnb_4bit_compute_dtype,
        )
        return quantize_config

    def get_lora_target(self, lora_target_config):
        return lora_target_config.targets

    def get_lora_config(self, lora_config, lora_targets: List):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=lora_targets,
            bias="none",
        )
        return lora_config

    def get_device(self):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        return device

    def get_train_args(self, train_args):
        args = TrainingArguments(
            output_dir=train_args.output_dir,
            run_name=train_args.run_name,
            logging_steps=train_args.log_interval,
            num_train_epochs=train_args.num_epochs,
            per_device_train_batch_size=train_args.batch_size,
            per_device_eval_batch_size=train_args.batch_size,
            gradient_accumulation_steps=train_args.gradient_accumulation_steps,
            dataloader_num_workers=train_args.num_workers,
            learning_rate=train_args.learning_rate,
            weight_decay=train_args.weight_decay,
            warmup_ratio=train_args.warmup_ratio,
            lr_scheduler_type=train_args.lr_scheduler,
            save_steps=train_args.save_ratio,
            eval_steps=train_args.eval_ratio,
            fp16=train_args.fp16,
            eval_strategy=train_args.eval_strategy,
            remove_unused_columns=train_args.remove_unused_columns,
            report_to=train_args.report_to,
        )
        return args

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
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            data_collator=data_collator,
        )
        return trainer

    def train(self, trainer):
        trainer.train()

    def save_model(self, model, save_path):
        model.save_pretrained(save_path, from_pt=True)

    def model_to_huggingface(self, model, model_id):
        model.push_to_hub(model_id)

    def tokenizer_to_huggingface(self, tokenizer, model_id):
        tokenizer.push_to_hub(model_id)
