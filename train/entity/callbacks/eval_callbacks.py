from transformers import TrainerCallback, TrainerControl, TrainerState


class GenerationEvalCallback(TrainerCallback):

    def __init__(self, eval_dataset, ignore_until_epoch=0):
        self.eval_dataset = eval_dataset
        self.ignore_until_epoch = ignore_until_epoch

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        pass
