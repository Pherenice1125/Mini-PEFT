import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.utils import is_torchdynamo_compiling

from .abstracts import LLMCache
from .config import LLMModelConfig

class DynamicCache(LLMCache):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif (
            len(self.key_cache[layer_idx]) == 0
        ):  # fills previously skipped layers; checking for tensor causes errors
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache)
            <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = (
            self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        )
        return layer_seq_length

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search.
        """
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx] != []:
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(
        self, full_batch_size: int, split_size: int
    ) -> List["DynamicCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [
                tensor[i : i + split_size] for tensor in self.key_cache
            ]
            current_split.value_cache = [
                tensor[i : i + split_size] for tensor in self.value_cache
            ]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["DynamicCache"]) -> "DynamicCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for idx in range(len(splits[0])):
            key_cache = [
                current.key_cache[idx]
                for current in splits
                if current.key_cache[idx] != []
            ]
            value_cache = [
                current.key_cache[idx]
                for current in splits
                if current.key_cache[idx] != []
            ]
            if key_cache != []:
                layer_keys = torch.cat(key_cache, dim=0)
                layer_values = torch.cat(value_cache, dim=0)
                cache.update(layer_keys, layer_values, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


class StaticCache(LLMCache):
    def __init__(
        self,
        config: LLMModelConfig,
        batch_size: int,
        max_cache_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.max_cache_len = (
            config.max_seq_len_ if max_cache_len is None else max_cache_len
        )

        self.head_dim = config.head_dim_

        self.dtype = dtype
        self.num_key_value_heads = config.n_kv_heads_

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        cache_shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.max_cache_len,
            self.head_dim,
        )
        for idx in range(config.n_layers_):
            new_layer_key_cache = torch.zeros(
                cache_shape, dtype=self.dtype, device=device
            )
            new_layer_value_cache = torch.zeros(
                cache_shape, dtype=self.dtype, device=device
            )
            # Notes:
            # 1. `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            #     breaks when updating the cache. It can't be used if the cache code is being compiled (but in that case
            #     it is not needed anyway)
            # 2. `torch.export()` requires mutations to be registered as buffers.
            if not is_torchdynamo_compiling():
                self.register_buffer(
                    f"key_cache_{idx}",
                    torch.zeros(cache_shape, dtype=dtype, device=device),
                )
                self.register_buffer(
                    f"value_cache_{idx}",
                    torch.zeros(cache_shape, dtype=dtype, device=device),
                )
                new_layer_key_cache = getattr(self, f"key_cache_{idx}")
                new_layer_value_cache = getattr(self, f"value_cache_{idx}")
                torch._dynamo.mark_static_address(new_layer_key_cache)
                torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)


cache_dict = {
    "static": StaticCache,
    "dynamic": DynamicCache,
}


def cache_factory(
        cache_implemtation: str,
        config: LLMModelConfig,
        batch_size: int,
        max_cache_len: int,
):
    assert(
        cache_implemtation in cache_dict # further options will be added later(dict)
    ), f"Unknown cache implementation: {cache_implemtation}"
    logging.info(f"Using cache implementation: {cache_implemtation}")
    if cache_implemtation == "sliding_window":
        assert hasattr(config, "sliding_window_")
        max_cache_len = min(config.sliding_window_, max_cache_len)
    return cache_dict[cache_implemtation](
        config=config,
        batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=config.device_,
        dtype=config.dtype_,
    )