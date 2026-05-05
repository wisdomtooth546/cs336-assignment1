from itertools import pairwise
import json
from typing import Iterable, Iterator
import regex as re


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self._byte_to_id = {b:i for i, b in vocab.items()}
        self._merges_as_ids = [(self._byte_to_id[a], self._byte_to_id[b])
                                for a,b in merges]
        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


    @staticmethod                                
    def _get_pairs(tokens):
        pairs = []
        for a, b in pairwise(tokens):
            pairs.append((a, b))
        return pairs
    
    @staticmethod
    def _merge_token(token_list, most_pair, token_id):
        new_token = []
        i = 0
        while i < len(token_list):
            if i+1 < len(token_list) and (token_list[i], token_list[i+1]) == most_pair:
                new_token.append(token_id)
                i += 2
            else:
                new_token.append(token_list[i])
                i += 1
        return new_token
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(merges_filepath, encoding="utf-8") as f:
            merges = [tuple(line.rstrip().split(" ")) for line in f]
        with open(vocab_filepath) as f:
            vocab = json.load(f)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _pretokenise(self, text: str):
        return re.finditer(self.pat, text)
    
    def _encode_str_to_id(self, pretoken_str):
        ids = []
        for i in pretoken_str.encode():
            ids.append(self._byte_to_id[bytes([i])])
        return ids

    def _get_next_merge(self, pairs):
        merge_candidates = [m for m in self._merges_as_ids if m in pairs]
        if not merge_candidates:
            return
        return min(merge_candidates, key=lambda x: self._merges_as_ids.index(x))
    
    def _encode_pretoken(self, pretoken_str,):
        pretoken_ids = self._encode_str_to_id(pretoken_str)
        pairs = self._get_pairs(pretoken_ids)
        merge = self._get_next_merge(pairs)
        while merge:
            a, b = merge
            token_id = self._byte_to_id[self.vocab[a] + self.vocab[b]]
            pretoken_ids = self._merge_token(pretoken_ids, merge, token_id)
            pairs = self._get_pairs(pretoken_ids)
            merge = self._get_next_merge(pairs)
        return pretoken_ids
    
    def encode(self, text: str) -> list[int]:
        parts = [text]
        if self.special_tokens:
            # Longest specials first so "<|endoftext|><|endoftext|>" matches before "<|endoftext|>".
            special_sorted = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(map(re.escape, special_sorted)) + ")"
            parts = re.split(pattern, text)

        special_set = set(self.special_tokens) if self.special_tokens else set()
        tokens: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in special_set:
                tokens.append(self._byte_to_id[part.encode("utf-8")])
            else:
                for regex_match in self._pretokenise(part):
                    pretoken_string = regex_match.group()
                    _tokens = self._encode_pretoken(pretoken_string)
                    tokens.extend(_tokens)
        return tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, tokens: list[int]) -> str:
        while any([isinstance(i, (int, float)) for i in tokens]):
            for i, token in enumerate(tokens):
                tokens[i] = self.vocab.get(token, token)
        return b"".join(tokens).decode("utf-8", errors="replace")  # pyright: ignore[reportArgumentType]