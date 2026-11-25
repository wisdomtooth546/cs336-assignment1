
from tqdm.auto import tqdm
import numpy as np
import torch
from torch import nn
from pydantic import BaseModel
import wandb

from model.checkpoint import load_checkpoint, save_checkpoint
from model.clipping import clip_gradients
from model.dataloader import get_batch
from model.loss import cross_entropy_loss
from model.optimiser import AdamW
from model.scheduler import cosine_lr
from model.transformer import PostNormTransformerLM, TransformerLM


class TrainingConfig(BaseModel):
    data_path: str | None
    validation_data_path: str | None
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    d_ff: int
    rope_theta: float
    num_heads: int
    lr_min: float
    lr_max: float = 3e-3
    lr_warmup: int
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 4
    resume_path: str | None = None
    epochs: int
    max_gradient_norm: float = 1.0
    log_interval: int = 5
    validate_and_save_interval: int = 10
    validation_epochs: int = 2
    total_train_tokens: int = 40_960_000
    total_validation_tokens: int = 20_960_000

@torch.inference_mode()
def validate(
    model: nn.Module,
    args: TrainingConfig,
):
    model.eval()

    data = np.memmap(
        args.validation_data_path,
        mode="r",
        dtype=np.uint16,
    )

    running_loss = 0.0
    global_step = 0

    val_epochs = tqdm(
        range(args.validation_epochs),
        desc="Validation",
        leave=False,
        dynamic_ncols=True,
    )
    steps_per_epoch = args.total_validation_tokens // (
        args.batch_size * args.context_length
    )

    for epoch in val_epochs:
        epoch_loss = 0.0

        val_steps = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch + 1}/{args.validation_epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        for step in val_steps:
            x, y = get_batch(
                data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
            )

            logits = model(x)
            loss = cross_entropy_loss(logits, targets=y)

            loss_value = loss.item()

            epoch_loss += loss_value
            running_loss += loss_value
            global_step += 1

            val_steps.set_postfix(
                loss=f"{loss_value:.4f}",
                avg_loss=f"{epoch_loss / (step + 1):.4f}",
            )

        epoch_avg_loss = epoch_loss / steps_per_epoch

        val_epochs.set_postfix(
            epoch_loss=f"{epoch_avg_loss:.4f}",
            avg_loss=f"{running_loss / global_step:.4f}",
        )

    model.train()

    return running_loss / global_step


def train(args: TrainingConfig):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    project = "assignment-1-postnorm"

    data = np.memmap(
        args.data_path,
        mode="r",
        dtype=np.uint16,
    )

    model = PostNormTransformerLM(
        args.vocab_size,
        args.context_length,
        args.num_layers,
        args.d_model,
        args.d_ff,
        args.rope_theta,
        args.num_heads,
        args.device,
    )

    optimiser = AdamW(
        model.parameters(),
        lr=args.lr_max,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    global_step = 0

    steps_per_epoch = args.total_train_tokens // (
        args.batch_size * args.context_length
    )

    total_steps = args.epochs * steps_per_epoch

    start_epoch = 0
    step_in_epoch = 0

    if args.resume_path:
        global_step = load_checkpoint(
            args.resume_path,
            model,
            optimiser,
        )

        start_epoch = global_step // steps_per_epoch
        step_in_epoch = global_step % steps_per_epoch

    best_validation_loss = float("inf")

    with wandb.init(project=project, config=args.model_dump()) as run:

        epoch_bar = tqdm(
            range(start_epoch, args.epochs),
            desc="Epochs",
            position=0,
            dynamic_ncols=True,
        )

        for epoch in epoch_bar:

            model.train()
            epoch_losses = []

            train_bar = tqdm(
                range(step_in_epoch, steps_per_epoch),
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                position=1,
                leave=False,
                dynamic_ncols=True,
            )

            for step in train_bar:

                x, y = get_batch(
                    data,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    device=args.device,
                )

                optimiser.zero_grad()

                # Learning rate schedule
                lr = cosine_lr(
                    t=global_step,
                    lr_min=args.lr_min,
                    lr_max=args.lr_max,
                    warmup_steps=args.lr_warmup,
                    total_steps=total_steps,
                )

                for param_group in optimiser.param_groups:
                    param_group["lr"] = lr

                # Forward pass
                logits = model(x)
                loss = cross_entropy_loss(
                    logits,
                    targets=y,
                )

                # Backward pass
                loss.backward()

                # Gradient clipping
                grad_norm = clip_gradients(
                    model.parameters(),
                    args.max_gradient_norm,
                    eps=args.eps,
                    device=args.device,
                )

                # Optimizer step
                optimiser.step()

                loss_value = loss.item()
                epoch_losses.append(loss_value)

                global_step += 1

                # W&B logging
                if global_step % args.log_interval == 0:
                    run.log({
                        "loss": loss_value,
                        "learning_rate": lr,
                        "grad_norm": (
                            grad_norm.item()
                            if isinstance(grad_norm, torch.Tensor)
                            else grad_norm
                        ),
                        "step": global_step,
                    })

                # Progress bar
                train_bar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    lr=f"{lr:.2e}",
                )

                # Validation and checkpointing
                if global_step % args.validate_and_save_interval == 0:

                    validation_loss = validate(
                        model,
                        args,
                    )

                    run.log({
                        "validation/loss": validation_loss,
                        "step": global_step,
                    })

                    train_bar.set_postfix(
                        loss=f"{loss_value:.4f}",
                        val_loss=f"{validation_loss:.4f}",
                        lr=f"{lr:.2e}",
                    )

                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss

                        save_checkpoint(
                            model,
                            optimiser,
                            global_step,
                            f"{project}-transformer_assignment_1.pt",
                        )

                        train_bar.write(
                            f"New best validation loss: "
                            f"{best_validation_loss:.4f} "
                            f"at step {global_step}. Checkpoint saved."
                        )

            step_in_epoch = 0

            epoch_loss = np.mean(epoch_losses)

            run.log({
                "epoch": epoch + 1,
                "train/epoch_loss": epoch_loss,
            })

            epoch_bar.set_postfix(
                train_loss=f"{epoch_loss:.4f}",
                best_val_loss=f"{best_validation_loss:.4f}",
            )
    save_checkpoint(
        model,
        optimiser,
        global_step,
        f"{project}-transformer_assignment_1_final.pt",
    )


if __name__ == "__main__":
    args = TrainingConfig(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        d_ff=1344,
        rope_theta=10000,
        num_heads=16,
        num_layers=4,
        validate_and_save_interval=1800,
        validation_epochs=1,
        lr_max=3e-4,
        lr_min=3e-5,
        lr_warmup=1000,
        batch_size=32,
        data_path="data/tinystories-train.bin",
        validation_data_path="data/TinyStories-valid.bin",
        epochs=5,
    )

    train(args)
