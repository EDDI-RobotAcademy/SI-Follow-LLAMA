from datasets import load_dataset, concatenate_datasets
import torch

from dataset.entity.data_collator import DataCollatorForSupervisedDataset
from dataset.repository.dataset_repository import DatasetRepository


class DatasetRepositoryImpl(DatasetRepository):
    IGNORE_INDEX = -100
    PROMPT_TEMPLATE = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: "
        )
    MAX_SEQ_LEN = 8192
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

    def load_dataset(self, dataset_id, tokenizer):
        def tokenization(examples):
            sources = []
            targets = []
            prompt = self.PROMPT_TEMPLATE
            for instruction, input, output in zip(examples['instruction'],examples['input'],examples['output']):
                if input is not None and input !="":
                    instruction = instruction+'\n'+input
                source = prompt.format_map({'instruction':instruction})
                target = f"{output}{tokenizer.eos_token}"

                sources.append(source)
                targets.append(target)

            tokenized_sources = tokenizer(sources,return_attention_mask=False)
            tokenized_targets = tokenizer(targets,return_attention_mask=False,add_special_tokens=False)

            all_input_ids = []
            all_labels = []
            for s,t in zip(tokenized_sources['input_ids'],tokenized_targets['input_ids']):
                input_ids = torch.LongTensor(s + t)[:self.MAX_SEQ_LEN]
                labels = torch.LongTensor([-100] * len(s) + t)[:self.MAX_SEQ_LEN]
                assert len(input_ids) == len(labels)
                all_input_ids.append(input_ids)
                all_labels.append(labels)

            results = {'input_ids':all_input_ids, 'labels': all_labels}
            return results

        all_datasets = []

        raw_dataset = load_dataset(dataset_id)
        tokenization_func = tokenization
        tokenized_dataset = raw_dataset.map(
            tokenization_func,
            batched=True,
            remove_columns=["instruction","input","output"],
            keep_in_memory=False,
            desc="preprocessing on dataset",
        )
        processed_dataset = tokenized_dataset
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
        all_datasets = concatenate_datasets(all_datasets)
        return all_datasets
    
    def get_data_collator(self, tokenizer):
        collator = DataCollatorForSupervisedDataset(tokenizer)
        return collator