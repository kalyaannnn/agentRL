"""Block-aligned prefix cache for continuous rollout prefill reuse."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class PrefixCacheBlock:
    """A cached block payload returned by lookup."""

    block_hash: str
    tokens: tuple[int, ...]
    kv_block: tuple[tuple[torch.Tensor, ...], ...]
    end_logits: torch.Tensor | None
    byte_size: int


@dataclass(frozen=True, slots=True)
class PrefixCacheHandle:
    """Retains cache entries while a caller reconstructs a prefix."""

    block_hashes: tuple[str, ...]


@dataclass(slots=True)
class _PrefixCacheEntry:
    block: PrefixCacheBlock
    refcount: int
    last_access: int


class PrefixCache:
    """Rolling-hash KV prefix cache keyed by full token blocks."""

    def __init__(self, block_size: int = 16, max_cache_bytes: int | None = None) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be > 0.")
        if max_cache_bytes is not None and max_cache_bytes < 0:
            raise ValueError("max_cache_bytes must be >= 0 when provided.")
        self.block_size = int(block_size)
        self.max_cache_bytes = max_cache_bytes
        self._entries: dict[str, _PrefixCacheEntry] = {}
        self._clock = 0
        self._total_bytes = 0
        self.eviction_count = 0

    @property
    def size_blocks(self) -> int:
        """Return the number of resident KV blocks."""

        return len(self._entries)

    @property
    def total_bytes(self) -> int:
        """Return the estimated resident bytes."""

        return self._total_bytes

    def block_hashes(self, token_ids: torch.Tensor | list[int] | tuple[int, ...]) -> tuple[str, ...]:
        """Return rolling hashes for every full block in `token_ids`."""

        tokens = self._tokens_tuple(token_ids)
        previous = "0" * 32
        hashes: list[str] = []
        for block in self._full_token_blocks(tokens):
            previous = self._hash_block(previous, block)
            hashes.append(previous)
        return tuple(hashes)

    def lookup(
        self,
        token_ids: torch.Tensor | list[int] | tuple[int, ...],
    ) -> tuple[list[PrefixCacheBlock], int, PrefixCacheHandle]:
        """Return the longest cached block-aligned prefix for `token_ids`."""

        tokens = self._tokens_tuple(token_ids)
        matched: list[PrefixCacheBlock] = []
        matched_hashes: list[str] = []
        previous = "0" * 32
        for block_tokens in self._full_token_blocks(tokens):
            block_hash = self._hash_block(previous, block_tokens)
            entry = self._entries.get(block_hash)
            if entry is None or entry.block.tokens != block_tokens:
                break
            self._touch(entry)
            entry.refcount += 1
            matched.append(entry.block)
            matched_hashes.append(block_hash)
            previous = block_hash

        handle = PrefixCacheHandle(block_hashes=tuple(matched_hashes))
        return matched, len(matched) * self.block_size, handle

    def put(
        self,
        token_ids: torch.Tensor | list[int] | tuple[int, ...],
        kv_blocks: list[tuple[tuple[torch.Tensor, ...], ...]],
        logits_by_block: list[torch.Tensor | None] | None = None,
    ) -> None:
        """Insert full token blocks and their KV payloads."""

        tokens = self._tokens_tuple(token_ids)
        token_blocks = self._full_token_blocks(tokens)
        if len(kv_blocks) != len(token_blocks):
            raise ValueError(
                f"kv_blocks length {len(kv_blocks)} does not match full token blocks {len(token_blocks)}."
            )
        if logits_by_block is not None and len(logits_by_block) != len(token_blocks):
            raise ValueError(
                f"logits_by_block length {len(logits_by_block)} does not match full token blocks {len(token_blocks)}."
            )

        previous = "0" * 32
        for index, (block_tokens, kv_block) in enumerate(zip(token_blocks, kv_blocks, strict=True)):
            block_hash = self._hash_block(previous, block_tokens)
            existing = self._entries.get(block_hash)
            if existing is not None and existing.block.tokens != block_tokens:
                raise ValueError("Prefix cache hash collision for different token IDs.")

            end_logits = None if logits_by_block is None else logits_by_block[index]
            byte_size = self._estimate_block_bytes(kv_block)
            block = PrefixCacheBlock(
                block_hash=block_hash,
                tokens=block_tokens,
                kv_block=kv_block,
                end_logits=end_logits,
                byte_size=byte_size,
            )
            self._clock += 1
            if existing is None:
                self._entries[block_hash] = _PrefixCacheEntry(
                    block=block,
                    refcount=0,
                    last_access=self._clock,
                )
                self._total_bytes += byte_size
            else:
                self._total_bytes += byte_size - existing.block.byte_size
                existing.block = block
                existing.last_access = self._clock
            previous = block_hash

        if self.max_cache_bytes is not None and self._total_bytes > self.max_cache_bytes:
            self.evict(self._total_bytes - self.max_cache_bytes)

    def release(self, handle: PrefixCacheHandle) -> None:
        """Release blocks retained by a prior lookup."""

        for block_hash in handle.block_hashes:
            entry = self._entries.get(block_hash)
            if entry is not None and entry.refcount > 0:
                entry.refcount -= 1

    def evict(self, target_bytes: int) -> int:
        """Evict unreferenced blocks until at least `target_bytes` are freed."""

        if target_bytes <= 0:
            return 0
        candidates = [
            (entry.refcount, entry.last_access, block_hash)
            for block_hash, entry in self._entries.items()
            if entry.refcount == 0
        ]
        candidates.sort()

        freed = 0
        evicted = 0
        for _refcount, _last_access, block_hash in candidates:
            entry = self._entries.pop(block_hash)
            self._total_bytes -= entry.block.byte_size
            freed += entry.block.byte_size
            evicted += 1
            if freed >= target_bytes:
                break
        self.eviction_count += evicted
        return evicted

    def contains(self, block_hash: str) -> bool:
        """Return whether a block hash is resident."""

        return block_hash in self._entries

    def refcount(self, block_hash: str) -> int:
        """Return the current refcount for a resident block."""

        entry = self._entries.get(block_hash)
        return 0 if entry is None else entry.refcount

    def _touch(self, entry: _PrefixCacheEntry) -> None:
        self._clock += 1
        entry.last_access = self._clock

    def _full_token_blocks(self, tokens: tuple[int, ...]) -> list[tuple[int, ...]]:
        full_length = (len(tokens) // self.block_size) * self.block_size
        return [
            tokens[start : start + self.block_size]
            for start in range(0, full_length, self.block_size)
        ]

    def _hash_block(self, previous_hash: str, block_tokens: tuple[int, ...]) -> str:
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(previous_hash.encode("ascii"))
        hasher.update(self.block_size.to_bytes(4, "little", signed=False))
        for token in block_tokens:
            hasher.update(int(token).to_bytes(8, "little", signed=True))
        return hasher.hexdigest()

    @staticmethod
    def _tokens_tuple(token_ids: torch.Tensor | list[int] | tuple[int, ...]) -> tuple[int, ...]:
        if isinstance(token_ids, torch.Tensor):
            return tuple(int(token) for token in token_ids.detach().cpu().view(-1).tolist())
        return tuple(int(token) for token in token_ids)

    @staticmethod
    def _estimate_block_bytes(kv_block: tuple[tuple[torch.Tensor, ...], ...]) -> int:
        tensor_bytes = sum(
            int(tensor.numel() * tensor.element_size())
            for layer in kv_block
            for tensor in layer
        )
        return max(32, tensor_bytes)
