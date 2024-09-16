# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import argparse

import evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    DataCollatorForSeq2Seq,
)

from accelerate import Accelerator, DistributedType

MAX_GPU_BATCH_SIZE = 1  # Decreased due to larger model size
EVAL_BATCH_SIZE = 1

def get_dataloaders(accelerator: Accelerator, tokenizer, model, batch_size: int = 1):
    """
    Creates DataLoaders for the samsum dataset, using the provided tokenizer and model.
    """
    datasets = load_dataset("samsum")

    def tokenize_function(examples):
        inputs = examples["dialogue"]
        targets = examples["summary"]
        model_inputs = tokenizer(
            inputs, max_length=1024, truncation=True, padding=False
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=128, truncation=True, padding=False
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=datasets["train"].column_names,
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8 if accelerator.mixed_precision != "no" else None,
    )

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=EVAL_BATCH_SIZE,
    )

    return train_dataloader, eval_dataloader

def training_function(config, args):
    # Initialize accelerator
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)
    # Sample hyperparameters
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    metric = evaluate.load("rouge")

    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.XLA:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    set_seed(seed)
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

    # Get dataloaders
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, tokenizer, model, batch_size)

    model = model.to(accelerator.device)
    optimizer = AdamW(params=model.parameters(), lr=lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=128,
                    num_beams=4,
                )
                labels = batch["labels"]
                labels = accelerator.pad_across_processes(
                    labels, dim=1, pad_index=tokenizer.pad_token_id
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )

                labels = accelerator.gather(labels)
                generated_tokens = accelerator.gather(generated_tokens)

                labels = labels.cpu().numpy()
                generated_tokens = generated_tokens.cpu().numpy()

                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )

                metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        eval_metric = metric.compute()
        accelerator.print(f"epoch {epoch}:", eval_metric)
    accelerator.end_training()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Flan-T5 on the samsum dataset.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 1}
    training_function(config, args)

if __name__ == "__main__":
    main()
