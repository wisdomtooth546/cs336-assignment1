from tokenizers import Tokenizer
import torch
from jaxtyping import Float
from torch import nn

from model.checkpoint import load_checkpoint
from model.layers.activations import softmax
from pydantic import BaseModel

from model.optimiser import AdamW
from model.train import TrainingConfig
from model.transformer import TransformerLM
from transformers import AutoTokenizer

class InferParams(BaseModel):
    max_tokens: int = 5
    temperature : int = 1
    top_p: float = 1

def get_next_token(
    logits: Float[torch.Tensor, "batch seq vocab"],
    temperature: float,
    top_p: float
    ):
    # Get the last element of the matrix
    y = logits[: , -1, :]
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    # Apply temperature first.
    logits = y / temperature

    # logits to probability
    probs = softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(
        probs, descending=True, dim=-1
    )

    to_remove = sorted_probs.cumsum(dim=-1) > top_p
    # Shift a position to the right
    to_remove[..., 1:] = to_remove[..., :-1].clone()
    # Set the first element to false so there is at least 1
    to_remove[..., 0] = False

    sorted_probs[to_remove] = 0.0

    updated_probs = torch.zeros_like(probs).scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_probs
    )
    probs = updated_probs / updated_probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)

def infer(
    model: nn.Module,
    tokenizer,
    prompt: str,
    args: InferParams,
    device: str | None = None
):
    x = tokenizer.encode(prompt).ids
    if device:
        model.to(device)
    x = torch.tensor(x, device=device).unsqueeze(0)  # (1, seq_length)
    model.eval()
    with torch.inference_mode():
        for i in range(args.max_tokens):
            logits = model(x)
            next_token = get_next_token(
                logits,
                temperature=args.temperature,
                top_p=args.top_p
            )
            if next_token == 0:
                break
            x = torch.concat([x, next_token], dim=-1)

    return tokenizer.decode(x[0].tolist())

if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    args = TrainingConfig(
        vocab_size=10000,
        data_path=None,
        validation_data_path=None,
        lr_min=1e-5,
        lr_max=1e-4,
        lr_warmup=500,
        context_length=256,
        d_model=512,
        d_ff=1344,
        rope_theta=10000,
        num_heads=16,
        num_layers=4,
        epochs=200,
        resume_path="transformer_assignment_1.pt"
    )
    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.num_layers,
        args.d_model,
        args.d_ff,
        args.rope_theta,
        args.num_heads,
        args.device
    )

    optimiser = AdamW(
        model.parameters(),
        lr=args.lr_max,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps
    )
    load_checkpoint(
        "/root/projects/assignment1-basics/transformer_assignment_1.pt",
        model,
        optimiser
    )
    infer_args = InferParams(
        max_tokens=256 
    )
    prompt = "Once upon a time,"
    y = infer(
        model,
        tokenizer,
        prompt,
        infer_args,
        device="cuda"
    )
    print(y)
