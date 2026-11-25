import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers import decoders, processors
import numpy as np
from pathlib import Path

VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]
DATASET = "data/TinyStories-valid.txt"


def train_tokenizer(save_path):
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # GPT-2 uses byte-level BPE
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    # Configure the BPE training algorithm
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    # Train on your corpus
    tokenizer.train(
        files=["TinyStories-train.txt"],
        trainer=trainer,
    )

    # Configure the tokenizer to behave like GPT-2
    tokenizer.post_processor = processors.ByteLevel(
        trim_offsets=False
    )

    tokenizer.decoder = decoders.ByteLevel()

    # Save the tokenizer
    tokenizer.save(save_path)

    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    return tokenizer

def save_tokens():

    input_path = Path(DATASET)
    output_path = input_path.with_suffix(".bin")
    tokenizer = Tokenizer.from_file("data/tokenizer.json")

    eot_id = tokenizer.token_to_id("<|endoftext|>")

    num_tokens = 0

    with open(input_path, "r", encoding="utf-8") as f:
        print(f"Saving tokens to {output_path}")
        with open(output_path, "wb") as out:
            for line in f:
                if not line.strip():
                    continue

                # Tokenize only one story at a time
                ids = tokenizer.encode(line).ids

                # Add end-of-story token
                ids.append(eot_id)

                # Convert to uint16 and write directly to disk
                chunk = np.asarray(ids, dtype=np.uint16)
                chunk.tofile(out)

                num_tokens += len(chunk)

    print(f"Saved {num_tokens:,} tokens")


if __name__ == "__main__":
    # if not os.path.exists("data/tokenizer.json"):
    #     train_tokenizer("data/tokenizer.json")
    save_tokens()