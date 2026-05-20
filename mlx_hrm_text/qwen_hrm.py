from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def _dtype_from_name(name: str | None):
    if name is None:
        return None
    mapping = {
        "bfloat16": mx.bfloat16,
        "bf16": mx.bfloat16,
        "float16": mx.float16,
        "fp16": mx.float16,
        "float32": mx.float32,
        "fp32": mx.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}. Use bfloat16, float16, or float32.") from exc


def _rms_norm(x: mx.array, weight: mx.array | None, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, weight, eps)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


@dataclass
class QwenHrmConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool = True
    model_type: str = "qwen3"
    split_index: int | None = None
    H_cycles: int = 1
    L_cycles: int = 1
    alpha_l: float = 0.03
    alpha_h: float = 0.05
    beta_l: float = 0.01
    beta_h: float = 0.01
    update_mix_l: float = 1.0
    update_mix_h: float = 1.0
    refined_delta_scale: float = 1.0
    logit_blend: float = 0.2
    quantization: dict[str, Any] | None = None
    quantization_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QwenHrmConfig":
        hidden_size = int(data["hidden_size"])
        num_attention_heads = int(data["num_attention_heads"])
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=hidden_size,
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=int(data.get("num_key_value_heads", num_attention_heads)),
            head_dim=int(data.get("head_dim", hidden_size // num_attention_heads)),
            max_position_embeddings=int(data["max_position_embeddings"]),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            rope_theta=float(data.get("rope_theta", 1000000.0)),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
            model_type=str(data.get("model_type", "qwen3")),
            split_index=int(data["split_index"]) if data.get("split_index") is not None else None,
            H_cycles=int(data.get("H_cycles", 1)),
            L_cycles=int(data.get("L_cycles", 1)),
            alpha_l=float(data.get("alpha_l", 0.03)),
            alpha_h=float(data.get("alpha_h", 0.05)),
            beta_l=float(data.get("beta_l", 0.01)),
            beta_h=float(data.get("beta_h", 0.01)),
            update_mix_l=float(data.get("update_mix_l", 1.0)),
            update_mix_h=float(data.get("update_mix_h", 1.0)),
            refined_delta_scale=float(data.get("refined_delta_scale", 1.0)),
            logit_blend=float(data.get("logit_blend", 0.2)),
            quantization=data.get("quantization"),
            quantization_config=data.get("quantization_config"),
        )

    @property
    def lower_layers(self) -> int:
        return self.split_index or (self.num_hidden_layers // 2)


class KVCache:
    def __init__(self, max_length: int | None = None):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.max_length = max_length
        self.offset = 0

    def update(self, key: mx.array, value: mx.array) -> tuple[mx.array, mx.array]:
        if self.max_length is not None:
            if self.offset + key.shape[2] > self.max_length:
                raise ValueError(f"KV cache capacity exceeded: {self.offset + key.shape[2]} > {self.max_length}")
            if self.keys is None:
                shape = (key.shape[0], key.shape[1], self.max_length, key.shape[3])
                self.keys = mx.zeros(shape, dtype=key.dtype)
                self.values = mx.zeros(shape, dtype=value.dtype)
            self.keys = mx.slice_update(self.keys, key, start_indices=mx.array(self.offset), axes=(2,))
            self.values = mx.slice_update(self.values, value, start_indices=mx.array(self.offset), axes=(2,))
            self.offset += key.shape[2]
            assert self.values is not None
            return (
                mx.slice(
                    self.keys,
                    start_indices=mx.array(0),
                    axes=(2,),
                    slice_size=(key.shape[0], key.shape[1], self.offset, key.shape[3]),
                ),
                mx.slice(
                    self.values,
                    start_indices=mx.array(0),
                    axes=(2,),
                    slice_size=(value.shape[0], value.shape[1], self.offset, value.shape[3]),
                ),
            )

        if self.keys is None:
            self.keys = key
            self.values = value
        else:
            self.keys = mx.concatenate([self.keys, key], axis=2)
            self.values = mx.concatenate([self.values, value], axis=2)
        return self.keys, self.values


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return _rms_norm(x, self.weight, self.eps)


class QwenAttention(nn.Module):
    def __init__(self, config: QwenHrmConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        position_offset: int | mx.array | None,
        cache: KVCache | None = None,
    ) -> mx.array:
        batch, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = mx.reshape(query, (batch, seq_len, self.num_heads, self.head_dim))
        key = mx.reshape(key, (batch, seq_len, self.num_key_value_heads, self.head_dim))
        value = mx.reshape(value, (batch, seq_len, self.num_key_value_heads, self.head_dim))

        query = self.q_norm(query)
        key = self.k_norm(key)

        query = mx.transpose(query, (0, 2, 1, 3))
        key = mx.transpose(key, (0, 2, 1, 3))
        value = mx.transpose(value, (0, 2, 1, 3))

        if position_offset is not None:
            query = mx.fast.rope(
                query,
                self.head_dim,
                traditional=False,
                base=self.rope_theta,
                scale=1.0,
                offset=position_offset,
            )
            key = mx.fast.rope(
                key,
                self.head_dim,
                traditional=False,
                base=self.rope_theta,
                scale=1.0,
                offset=position_offset,
            )

        if cache is not None:
            cache_offset = cache.offset
            key, value = cache.update(key, value)
            mask = "causal" if cache_offset == 0 and seq_len > 1 else None
        else:
            mask = "causal"

        attn = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.scale, mask=mask)
        attn = mx.transpose(attn, (0, 2, 1, 3))
        attn = mx.reshape(attn, (batch, seq_len, self.num_heads * self.head_dim))
        return self.o_proj(attn)


class QwenMLP(nn.Module):
    def __init__(self, config: QwenHrmConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_silu(self.gate_proj(x)) * self.up_proj(x))


class QwenDecoderLayer(nn.Module):
    def __init__(self, config: QwenHrmConfig):
        super().__init__()
        self.self_attn = QwenAttention(config)
        self.mlp = QwenMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(self, x: mx.array, *, position_offset: int | mx.array | None, cache: KVCache | None = None) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = residual + self.self_attn(x, position_offset=position_offset, cache=cache)

        residual = x
        x = self.post_attention_layernorm(x)
        return residual + self.mlp(x)


class QwenHrmBackbone(nn.Module):
    def __init__(self, config: QwenHrmConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [QwenDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles
        self.alpha_l = config.alpha_l
        self.alpha_h = config.alpha_h
        self.beta_l = config.beta_l
        self.beta_h = config.beta_h
        self.update_mix_l = config.update_mix_l
        self.update_mix_h = config.update_mix_h
        self.refined_delta_scale = config.refined_delta_scale

    @property
    def lower(self) -> list[QwenDecoderLayer]:
        return self.layers[: self.config.lower_layers]

    @property
    def upper(self) -> list[QwenDecoderLayer]:
        return self.layers[self.config.lower_layers :]

    def make_cache(self, max_length: int | None = None) -> dict[str, Any]:
        return {
            "base": [KVCache(max_length=max_length) for _ in self.layers],
            "warm_L": [KVCache(max_length=max_length) for _ in self.lower],
            "warm_H": [KVCache(max_length=max_length) for _ in self.upper],
            "L": [
                [KVCache(max_length=max_length) for _ in self.lower]
                for _ in range(max(0, self.H_cycles * self.L_cycles))
            ],
            "H": [[KVCache(max_length=max_length) for _ in self.upper] for _ in range(max(0, self.H_cycles))],
        }

    def _run_layers(
        self,
        layers: list[QwenDecoderLayer],
        x: mx.array,
        *,
        position_offset: int | mx.array | None,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        for idx, layer in enumerate(layers):
            x = layer(x, position_offset=position_offset, cache=None if cache is None else cache[idx])
        return x

    def __call__(
        self,
        input_ids: mx.array,
        *,
        position_offset: int | mx.array | None,
        cache: dict[str, Any] | None = None,
    ) -> tuple[mx.array, mx.array]:
        x_embed = self.embed_tokens(input_ids)

        base = self._run_layers(
            self.layers,
            x_embed,
            position_offset=position_offset,
            cache=None if cache is None else cache["base"],
        )
        base = self.norm(base)

        if self.H_cycles <= 0 or self.L_cycles <= 0:
            return base, base

        z_l = self._run_layers(
            self.lower,
            x_embed,
            position_offset=position_offset,
            cache=None if cache is None else cache["warm_L"],
        )
        z_h = self._run_layers(
            self.upper,
            z_l,
            position_offset=position_offset,
            cache=None if cache is None else cache["warm_H"],
        )

        l_pass = 0
        for h_idx in range(self.H_cycles):
            for _ in range(self.L_cycles):
                mixed_l = _rms_norm(
                    z_l + self.alpha_l * z_h + self.beta_l * x_embed,
                    None,
                    self.config.rms_norm_eps,
                )
                next_l = self._run_layers(
                    self.lower,
                    mixed_l,
                    position_offset=position_offset,
                    cache=None if cache is None else cache["L"][l_pass],
                )
                z_l = (1.0 - self.update_mix_l) * z_l + self.update_mix_l * next_l
                l_pass += 1

            mixed_h = _rms_norm(
                z_h + self.alpha_h * z_l + self.beta_h * x_embed,
                None,
                self.config.rms_norm_eps,
            )
            next_h = self._run_layers(
                self.upper,
                mixed_h,
                position_offset=position_offset,
                cache=None if cache is None else cache["H"][h_idx],
            )
            z_h = (1.0 - self.update_mix_h) * z_h + self.update_mix_h * next_h

        refined = base + self.refined_delta_scale * (self.norm(z_h) - base)
        return base, refined


class QwenHrmForCausalLM(nn.Module):
    def __init__(self, config: QwenHrmConfig):
        super().__init__()
        self.config = config
        self.model = QwenHrmBackbone(config)
        self.lm_head = None if config.tie_word_embeddings else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logit_blend = config.logit_blend
        self.use_static_cache = False

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        *,
        dtype: str | None = None,
        strict: bool = True,
        revision: str | None = None,
    ) -> "QwenHrmForCausalLM":
        model_path = Path(model_id_or_path)
        if not model_path.exists():
            from huggingface_hub import snapshot_download

            model_path = Path(
                snapshot_download(
                    str(model_id_or_path),
                    revision=revision,
                    allow_patterns=[
                        "config.json",
                        "*.safetensors",
                        "*.safetensors.index.json",
                        "tokenizer*",
                        "vocab.json",
                        "merges.txt",
                        "special_tokens_map.json",
                        "generation_config.json",
                    ],
                )
            )

        config_data = json.loads((model_path / "config.json").read_text())
        config = QwenHrmConfig.from_dict(config_data)
        model = cls(config)

        quantization = config.quantization or config.quantization_config
        if quantization is not None:
            nn.quantize(
                model,
                bits=int(quantization.get("bits", 4)),
                group_size=int(quantization.get("group_size", 64)),
                mode=str(quantization.get("mode", "affine")),
            )

        weights = model_path / "model.safetensors"
        if not weights.exists():
            safetensors = sorted(model_path.glob("*.safetensors"))
            if len(safetensors) != 1:
                raise FileNotFoundError(f"Expected one safetensors file in {model_path}, found {len(safetensors)}")
            weights = safetensors[0]
        model.load_weights(str(weights), strict=strict)

        target_dtype = _dtype_from_name(dtype)
        if target_dtype is not None:
            model.set_dtype(target_dtype)
        return model

    def make_cache(self, max_length: int | None = None) -> dict[str, Any]:
        return self.model.make_cache(max_length=max_length)

    def _project_logits(self, hidden: mx.array) -> mx.array:
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return self.model.embed_tokens.as_linear(hidden)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        position_ids: mx.array | None = None,
        cache: dict[str, Any] | None = None,
    ) -> mx.array:
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        if position_ids is None:
            position_offset = 0
        else:
            if position_ids.ndim == 1:
                position_ids = position_ids[None, :]
            position_offset = position_ids[:, 0] if position_ids.ndim == 2 else position_ids[0]

        base_hidden, refined_hidden = self.model(input_ids, position_offset=position_offset, cache=cache)
        base_logits = self._project_logits(base_hidden)
        if self.logit_blend <= 0:
            return base_logits

        refined_logits = self._project_logits(refined_hidden)
        if self.logit_blend >= 1:
            return refined_logits
        return (1.0 - self.logit_blend) * base_logits + self.logit_blend * refined_logits

    def prefill(self, input_ids: mx.array, cache: dict[str, Any] | None = None) -> mx.array:
        if cache is None:
            cache = self.make_cache()
        logits = self(input_ids[None, :] if input_ids.ndim == 1 else input_ids, cache=cache)
        mx.eval(logits)
        return logits[:, -1, :]

    def decode_one(self, input_ids: mx.array, position: int, cache: dict[str, Any]) -> mx.array:
        if input_ids.ndim == 0:
            input_ids = input_ids[None, None]
        elif input_ids.ndim == 1:
            input_ids = input_ids[:, None]
        position_ids = mx.full(input_ids.shape, position, dtype=mx.int32)
        logits = self(input_ids, position_ids=position_ids, cache=cache)
        mx.eval(logits)
        return logits[:, -1, :]
