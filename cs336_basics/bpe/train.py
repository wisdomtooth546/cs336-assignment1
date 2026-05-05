import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
from typing import BinaryIO
from itertools import pairwise
from collections import defaultdict
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_pretoken_freq_table(pretokens: list[tuple[int]]):
    pretoken_freq = defaultdict(int)
    for pretoken in pretokens:
        pretoken_freq[pretoken] += 1
    return pretoken_freq


def pretokenise(text: str):
    return re.finditer(PAT, text)

def convert_to_utf_bytes(text: str):
    """ Convert to interger representation of utf encoded bytes """
    return tuple(text.encode())

def get_int_sequences(text: str, special_tokens):
    pattern = "|".join(map(re.escape, special_tokens))
    parts = re.split(pattern, text)
    parts = [p for p in parts if p.strip()]
    pretokens = []
    for part in parts:
        for regex_match in pretokenise(part):
            pretoken_string = regex_match.group()
            _pretokens = convert_to_utf_bytes(pretoken_string)
            pretokens.append(_pretokens)
    return pretokens


def process_chunk(args):
    file_path, start, end, special_tokens = args
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    pretokens = get_int_sequences(chunk, special_tokens)
    return pretokens

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

from itertools import pairwise

def get_byte_pair_count(pretoken_freq_table):
    byte_pairs = defaultdict(int)
    for pretoken in pretoken_freq_table:
        for a, b in pairwise(pretoken):
            byte_pairs[(a,b)] += pretoken_freq_table[pretoken]
    return byte_pairs

def merge_tokens(freq_table, most_pair: tuple[int, int], new_id):
    pretokens = list(freq_table)
    for pretoken in pretokens:
        new_token = []
        i = 0
        while i < len(pretoken):
            if i+1 < len(pretoken) and (pretoken[i], pretoken[i+1]) == most_pair:
                new_token.append(new_id)
                i += 2
            else:
                new_token.append(pretoken[i])
                i += 1
        new_token = tuple(new_token)
        if new_token != pretoken:
            freq_table[new_token] = freq_table[pretoken]
            freq_table.pop(pretoken)

    return freq_table

def get_most_common_pair(pairs: dict, vocab: dict):
    max_pair = max(pairs, key=lambda p: (pairs[p], (vocab[p[0]], vocab[p[1]])))
    return max_pair


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Learn BPE merges by keeping the corpus as lists of token ids and a global pair_counts map.
    After each merge, only pairs overlapping the merged (a,b) window change, so we patch
    pair_counts in O(1) per merge site instead of rescanning all pairs.
    """
    filepath = Path(input_path)
    if not filepath.exists():
        raise FileNotFoundError

    with open(filepath, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")


    vocab = {i: bytes([i]) for i in range(256)}
    for i, special_token in enumerate(special_tokens, start=max(vocab.keys()) + 1):
        vocab[i] = special_token.encode()
    

    chunks = [(filepath, start, end, special_tokens) for start, end in pairwise(boundaries)]
    with ThreadPoolExecutor() as pool:
        results = pool.map(process_chunk, chunks)
    
    pretokens = []
    for result in results:
        pretokens.extend(result)
    
    freq_table = get_pretoken_freq_table(pretokens=pretokens)
    merges: list[tuple[bytes, bytes]] = []
    next_merge_id = max(vocab.keys()) + 1

    while len(vocab) < vocab_size:
        byte_pairs = get_byte_pair_count(freq_table)
        if not byte_pairs:
            break
        most_pair = get_most_common_pair(byte_pairs, vocab)
        if byte_pairs[most_pair] <= 0:
            break
        most_pair = get_most_common_pair(byte_pairs, vocab)
        a, b = most_pair

        merges.append((vocab[a], vocab[b]))
        vocab[next_merge_id] = vocab[a] + vocab[b]
        freq_table = merge_tokens(freq_table, most_pair, next_merge_id)
        next_merge_id += 1

    return vocab, merges