from __future__ import annotations

import tqdm
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics import adam
from cs336_basics import cross_entropy_loss
from cs336_basics import get_batch
from cs336_basics import gradient_clipping
from cs336_basics import lr_cosine_schedule
from cs336_basics import save_checkpoint
from cs336_basics import transformer_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model on token IDs.")
    parser.add_argument("--train_data_path", type=Path, required=True, help="Path to a 1D .npy array of training token IDs.")
    parser.add_argument("--valid_data_path", type=Path, required=True, help="Path to a 1D .npy array of validation token IDs.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for checkpoints and training metrics.")

    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--theta", type=float, default=10_000.0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, required=True)
    parser.add_argument("--max_learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--resume_from", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_type", type=str, default = "pre")
    parser.add_argument("--no_rope", action="store_true", help="Whether to use ROPE or not.")
    return parser.parse_args()


def load_token_ids(path: str) -> np.ndarray:
    dataset = np.load(path, "r")
    if dataset.ndim !=1:
        raise ValueError(f"Expected a 1D array of token IDs, but got an array with shape {dataset.shape}.")
    return dataset

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    validation_data: np.ndarray,
    context_length: int,
    batch_size: int,
    device: torch.device,
    num_batches: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        inputs, targets = get_batch.get_batch(validation_data, context_length, batch_size, device)
        loggits = model(inputs)
        loss = cross_entropy_loss.cross_entropy(loggits, targets)
        losses.append(loss.item())
        # inputs, targets = get_batch.get_batch(validation_data, context_length, batch_size, device)
        # logits = model(inputs)
        # loss = cross_entropy_loss.cross_entropy(logits, targets)
        # losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train_model() -> None:
    args = parse_args()
    model_type = args.model_type
    no_rope: bool = args.no_rope
    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")
    if args.max_iters <= 0:
        raise ValueError("max_iters must be positive.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_data = load_token_ids(args.train_data_path)
    validation_data = load_token_ids(args.valid_data_path)
    if len(train_data) <= args.context_length or len(validation_data) <= args.context_length:
        raise ValueError("Each dataset must contain more tokens than context_length.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    if not args.no_rope:
        if model_type == "pre":
            model = transformer_lm.transformerlm(
                num_layers=args.num_layers,
                vocab_size=args.vocab_size,
                max_seq_length=args.context_length,
                d_ff=args.d_ff,
                d_model=args.d_model,
                num_heads=args.num_heads,
                theta=args.theta,
                device=device,
                dtype=torch.float32,
            ).to(device)
        elif model_type == "post":
            model = transformer_lm.posttransformerlm(
                num_layers=args.num_layers,
                vocab_size=args.vocab_size,
                max_seq_length=args.context_length,
                d_ff=args.d_ff,
                d_model=args.d_model,
                num_heads=args.num_heads,
                theta=args.theta,
                device=device,
                dtype=torch.float32,
            ).to(device)
    else:
        model = transformer_lm.transformerlm_no_rope(
                num_layers=args.num_layers,
                vocab_size=args.vocab_size,
                max_seq_length=args.context_length,
                d_ff=args.d_ff,
                d_model=args.d_model,
                num_heads=args.num_heads,
                theta=args.theta,
                device=device,
                dtype=torch.float32,
            ).to(device)
    parameters = list(model.parameters())
    optimizer = adam.AdamW(
        parameters,
        lr=args.max_learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume_from is not None:
        start_step = save_checkpoint.load_checkpoint(args.resume_from, model, optimizer)

    model.train()
    start_time = time.perf_counter()

    for step in tqdm.tqdm(range(start_step, args.max_iters)):
        learning_rate = lr_cosine_schedule.get_lr_cosine_schedule(
            step,
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_iters,
            args.max_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        inputs, targets = get_batch.get_batch(train_data, args.context_length, args.batch_size, device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy_loss.cross_entropy(logits, targets)
        loss.backward()
        if args.grad_clip > 0:
            gradient_clipping.gradient_clipping(parameters, args.grad_clip)
        optimizer.step()

        completed_step = step + 1
        should_evaluate = completed_step % args.eval_interval == 0 or completed_step == args.max_iters
        should_log = completed_step % args.log_interval == 0 or should_evaluate or completed_step == 1
        validation_loss = None
        if should_evaluate:
            validation_loss = evaluate(
                model,
                validation_data,
                args.context_length,
                args.batch_size,
                device,
                args.eval_batches,
            )

        if should_log:
            metrics = {
                "step": completed_step,
                "tokens_seen": completed_step * args.batch_size * args.context_length,
                "elapsed_seconds": time.perf_counter() - start_time,
                "learning_rate": learning_rate,
                "train_loss": loss.item(),
                "validation_loss": validation_loss,
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics))

        if args.checkpoint_interval > 0 and completed_step % args.checkpoint_interval == 0:
            save_checkpoint.save_checkpoint(
                model,
                optimizer,
                completed_step,
                args.output_dir / f"checkpoint_{completed_step}.pt",
            )

    save_checkpoint.save_checkpoint(model, optimizer, args.max_iters, args.output_dir / "checkpoint_final.pt")


# Original draft retained for reference:
# def train_model():
#     parser = argparse.ArgumentParser(description="Train a model on a given dataset")
#     parser.add_argument("--vocab_size", type=str, required=True, help="vocab size")
#     parser.add_argument("--context_length", type=str, required=True, help="context length")
#     args = parser.parse_args()
#     # Load the tokenizer
#     vocab_path = "tokenizer/vocab.json"
#     merges_path = "tokenizer/merges.json"
#     config_path = "tokenizer/config.json"
#     tokenizer = tokenization.Tokenizer.from_files(vocab_path, merges_path, config_path)
#     model = transformer_lm()
#     optimizer = adam.AdamW(model.parameters(), lr=1e-4)
#     for i in range():
#         inputs, target_y = get_batch()
#         y = model(inputs)
#         loss = cross_entropy_loss(y, target)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()


if __name__ == "__main__":
    train_model()
