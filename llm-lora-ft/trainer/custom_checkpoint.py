from transformers import TrainerCallback

class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, n, output_dir, tokenizer):
        self.n = n
        self.output_dir = output_dir
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)

        if epoch % self.n == 0:
            save_path = f"{self.output_dir}/checkpoint-epoch-{epoch}"

            model = kwargs["model"]
            model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

        return control