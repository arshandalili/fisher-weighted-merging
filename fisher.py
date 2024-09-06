from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import load_dataset
import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
import gc
from tqdm import tqdm


def compute_single_fisher(instance, model, trainable_params, num_labels=2):
    model.eval()
    instance_input_ids, instance_attention_mask = instance["input_ids"].to(
        device
    ), instance["attention_mask"].to(device)
    logits = model(
        input_ids=instance_input_ids, attention_mask=instance_attention_mask
    ).logits
    logits = logits.squeeze(0)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    sq_grads = []

    for i in range(num_labels):
        log_prob = log_probs[i]
        grads = torch.autograd.grad(
            log_prob, trainable_params, create_graph=False, retain_graph=True
        )
        sq_grad = [probs[i] * grad.pow(2) for grad in grads]
        sq_grads.append(sq_grad)
    with torch.no_grad():
        single_fisher = [
            torch.sum(torch.stack(grads_components), dim=0)
            for grads_components in zip(*sq_grads)
        ]
    del sq_grads, log_probs, logits, probs
    gc.collect()
    torch.cuda.empty_cache()

    return single_fisher


def tokenize_function(examples):
    return tokenizer(
        examples[args.text_col],
        add_special_tokens=True,
        return_token_type_ids=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="FacebookAI/roberta-base")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--task", type=str, default="imdb")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample_size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print(
        f"Calculating Fisher Information Matrix for {args.model} on {args.dataset} for {args.task} classification task."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Model Name/Path: {model.name_or_path}")

    train_ds = load_dataset(args.dataset, split=args.split)
    train_ds_size = len(train_ds)
    train_ds = train_ds.shuffle(seed=args.seed).select(range(args.sample_size))
    train_ds = train_ds.with_format("torch")
    tokenized_train_ds = train_ds.map(tokenize_function, batched=True)

    print(f"Dataset size: {train_ds_size}")

    trainable_params = [p for p in model.roberta.parameters() if p.requires_grad] + [
        p for p in model.classifier.parameters() if p.requires_grad
    ]
    fishers = [
        torch.Tensor(torch.zeros(p.shape, requires_grad=False, device=device))
        for p in trainable_params
    ]

    data_loader = DataLoader(tokenized_train_ds, batch_size=1)

    for batch in tqdm(data_loader):
        batch_fisher = compute_single_fisher(batch, model, trainable_params)
        fishers = [f_p + b_p for (f_p, b_p) in zip(fishers, batch_fisher)]

    fishers = [f * train_ds_size / len(data_loader) for f in fishers]

    new_model = AutoModelForSequenceClassification.from_pretrained(args.model).to(
        device
    )

    with torch.no_grad():
        new_model_params = [p for p in new_model.roberta.parameters()] + [
            p for p in new_model.classifier.parameters()
        ]
        for param, fisher in zip(new_model_params, fishers):
            if param.shape == fisher.shape:
                param.data.copy_(fisher)
            else:
                raise ValueError(
                    f"Shape mismatch: parameter {param.shape} vs fisher tensor {fisher.shape}"
                )

    output_dir = f"/ukp-storage-1/dalili/mergekit_test/fishers/{args.task}_sum"

    new_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Fisher Information Matrix (sum) saved at {output_dir}")
