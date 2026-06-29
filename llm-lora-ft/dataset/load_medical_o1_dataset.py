import torch


def get_medical_resoning_sft(dataset, tokenizer):

    # shuffle and split the dataset
    ds = dataset["train"].shuffle(seed=42)
    split = ds.train_test_split(test_size=0.1)

    train_ds = split["train"]
    eval_ds = split["test"]

    # def format - without chain of thought

    def format_example(example):
        messages = [
            {"role": "user", "content": example["Question"]},
            {"role": "assistant", "content": example["Response"]},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False
            )
        }

    # Apply formatting and remove unused columns for the datset

    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
    eval_ds  = eval_ds.map(format_example, remove_columns=eval_ds.column_names)

    return train_ds, eval_ds




