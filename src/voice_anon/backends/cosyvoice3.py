"""Backend-native CosyVoice3 batch primitives.

The upstream production example batches the LLM but currently loops over every
item for flow and HiFT.  These helpers preserve the official model equations
while padding variable token lengths into one real flow/vocoder batch.  Outputs
are cropped back to their individual valid lengths before they leave the
backend boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


COSYVOICE3_SPEAKER_DIMENSION = 192


@dataclass(frozen=True, slots=True)
class CosyVoice3BatchAudio:
    waveforms: tuple[Tensor, ...]
    sample_rate: int = 24_000


@torch.inference_mode()
def sample_speech_tokens_batch(
    model: torch.nn.Module,
    input_items: Sequence[Tensor],
    *,
    metadata: Mapping[str, object],
    seeds: Sequence[int],
    minimum_tokens: Sequence[int],
    maximum_tokens: Sequence[int],
    temperature: float = 0.8,
    top_k: int = 25,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    logit_quantization: float = 0.25,
    implementation: str = "fast",
    compact_interval: int = 32,
    decode_backend: str = "dynamic",
) -> list[list[int]]:
    """Sample a true LLM batch with per-item deterministic random streams.

    Hugging Face's shared generator consumes different random numbers when the
    batch shape changes.  Pre-generating one PCG64 stream per utterance makes
    batch and batch-one runs comparable while retaining this project's existing
    top-k/top-p parameters. This is not the upstream RAS sampler. BF16 kernels
    do not guarantee bit-exact seeded sequences across different batch shapes.
    """

    if implementation == "fast":
        return _sample_speech_tokens_fast(
            model, input_items, metadata=metadata, seeds=seeds,
            minimum_tokens=minimum_tokens, maximum_tokens=maximum_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, logit_quantization=logit_quantization,
            compact_interval=compact_interval,
            decode_backend=decode_backend,
        )
    if implementation != "legacy":
        raise ValueError("implementation must be fast or legacy")
    batch_size = len(input_items)
    if not (
        batch_size
        and len(seeds) == batch_size
        and len(minimum_tokens) == batch_size
        and len(maximum_tokens) == batch_size
    ):
        raise ValueError("CosyVoice3 LLM batch fields have incompatible sizes")
    if (
        temperature <= 0
        or top_k < 1
        or not 0 < top_p <= 1
        or logit_quantization <= 0
    ):
        raise ValueError("invalid sampling configuration")
    device = next(model.parameters()).device
    eos_id = int(metadata["eos_token_id"])
    speech_offset = int(metadata["speech_token_offset"])
    base_speech_tokens = int(metadata["base_speech_token_size"])
    embedding_size = int(metadata["embedding_size"])
    maximum_steps = max(int(value) for value in maximum_tokens)
    if maximum_steps < 1:
        raise ValueError("maximum token counts must be positive")
    random_values = np.stack(
        [
            np.random.Generator(np.random.PCG64(int(seed))).random(
                maximum_steps, dtype=np.float32
            )
            for seed in seeds
        ]
    )
    uniforms = torch.from_numpy(random_values).to(device)

    maximum_input = max(len(item) for item in input_items)
    padded, masks = [], []
    for item in input_items:
        missing = maximum_input - len(item)
        padded.append(torch.cat((torch.full((missing,), eos_id), item)))
        masks.append(torch.cat((torch.zeros(missing), torch.ones(len(item)))))
    initial_ids = torch.stack(padded).long().to(device)
    attention = torch.stack(masks).long().to(device)
    output = model(input_ids=initial_ids, attention_mask=attention, use_cache=True)
    cache = output.past_key_values
    logits = output.logits[:, -1].float()
    generated_global: list[Tensor] = []
    generated_speech: list[Tensor] = []
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    stop_steps = torch.full(
        (batch_size,), maximum_steps, dtype=torch.long, device=device
    )
    minimum = torch.as_tensor(minimum_tokens, device=device, dtype=torch.long)
    maximum = torch.as_tensor(maximum_tokens, device=device, dtype=torch.long)

    for step in range(maximum_steps):
        scores = logits / temperature
        # Match HF repetition penalty over both prompt and generated history.
        history = (
            torch.cat((initial_ids, torch.stack(generated_global, dim=1)), dim=1)
            if generated_global
            else initial_ids
        )
        for row in range(batch_size):
            indices = torch.unique(history[row])
            current = scores[row, indices]
            scores[row, indices] = torch.where(
                current < 0,
                current * repetition_penalty,
                current / repetition_penalty,
            )
        scores = torch.round(scores / logit_quantization) * logit_quantization
        allowed = scores[:, speech_offset : speech_offset + embedding_size]
        # Before the text-dependent minimum, only ordinary speech units 0..6560
        # are legal. Afterwards any CosyVoice3 speech special token terminates.
        before_minimum = step < minimum
        if before_minimum.any():
            allowed[before_minimum, base_speech_tokens:] = -torch.inf
        k = min(top_k, allowed.shape[1])
        values, local_indices = torch.topk(allowed, k=k, dim=-1)
        # Float64 over only top-k entries makes each row's CDF insensitive to
        # the CUDA reduction kernel selected for batch one versus batch N.
        probabilities = torch.softmax(values.double(), dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        remove = cumulative - probabilities > top_p
        probabilities = probabilities.masked_fill(remove, 0.0)
        probabilities /= probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cdf = probabilities.cumsum(dim=-1)
        chosen_position = (cdf < uniforms[:, step : step + 1]).sum(dim=-1)
        chosen_position.clamp_(max=k - 1)
        local_token = local_indices.gather(1, chosen_position[:, None]).squeeze(1)
        global_token = local_token + speech_offset

        forced_stop = step + 1 >= maximum
        stopping_special = local_token >= base_speech_tokens
        newly_special = (~finished) & stopping_special
        newly_forced = (~finished) & (~stopping_special) & forced_stop
        stop_steps = torch.where(newly_special, step, stop_steps)
        stop_steps = torch.where(newly_forced, step + 1, stop_steps)
        stored_speech = torch.where(
            finished | stopping_special,
            torch.zeros_like(local_token),
            local_token,
        )
        global_token = torch.where(finished, torch.full_like(global_token, eos_id), global_token)
        generated_global.append(global_token)
        generated_speech.append(stored_speech)
        finished |= stopping_special | forced_stop
        if bool(finished.all()):
            break

        attention = torch.cat(
            (attention, (~finished).long().unsqueeze(1)), dim=1
        )
        output = model(
            input_ids=global_token.unsqueeze(1),
            attention_mask=attention,
            past_key_values=cache,
            use_cache=True,
        )
        cache = output.past_key_values
        logits = output.logits[:, -1].float()

    values = torch.stack(generated_speech, dim=1).cpu()
    lengths = stop_steps.cpu().tolist()
    return [values[row, : int(length)].tolist() for row, length in enumerate(lengths)]


@torch.inference_mode()
def _sample_speech_tokens_fast(
    model, input_items, *, metadata, seeds, minimum_tokens, maximum_tokens,
    temperature, top_k, top_p, repetition_penalty, logit_quantization,
    compact_interval,
    decode_backend,
):
    """Same row-wise sampling rules, speech-only head and vectorized history.

    Explicit position IDs exclude left padding. No prompt truncation, changed
    sampling hyperparameters, or shared RNG streams are used for throughput.
    The legacy implementation remains available for numerical regression checks.
    """
    batch_size = len(input_items)
    if not (batch_size and len(seeds) == len(minimum_tokens) == len(maximum_tokens) == batch_size):
        raise ValueError("CosyVoice3 LLM batch fields have incompatible sizes")
    if temperature <= 0 or top_k < 1 or not 0 < top_p <= 1 or logit_quantization <= 0:
        raise ValueError("invalid sampling configuration")
    if any(len(item) < 1 for item in input_items) or min(maximum_tokens) < 1:
        raise ValueError("input and maximum token counts must be positive")
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise TypeError("fast sampler requires the pinned Qwen2 causal LM")
    if decode_backend not in {"dynamic", "cudagraph"}:
        raise ValueError("decode backend must be dynamic or cudagraph")
    device = next(model.parameters()).device
    eos_id = int(metadata["eos_token_id"])
    offset = int(metadata["speech_token_offset"])
    size = int(metadata["embedding_size"])
    base = int(metadata["base_speech_token_size"])
    steps = max(int(n) for n in maximum_tokens)
    uniforms = torch.from_numpy(np.stack([
        np.random.Generator(np.random.PCG64(int(seed))).random(steps, dtype=np.float32)
        for seed in seeds
    ])).to(device)
    lengths = torch.tensor([len(x) for x in input_items], device=device)
    width = max(len(x) for x in input_items)
    ids = torch.full((batch_size, width), eos_id, dtype=torch.long, device=device)
    attention = torch.zeros((batch_size, width + steps), dtype=torch.long, device=device)
    seen = torch.zeros((batch_size, size), dtype=torch.bool, device=device)
    for row, item in enumerate(input_items):
        item = item.to(device=device, dtype=torch.long)
        ids[row, -len(item):] = item
        attention[row, width-len(item):width] = 1
        local = item - offset
        seen[row, local[(local >= 0) & (local < size)]] = True
    positions = attention[:, :width].cumsum(-1) - 1
    positions.masked_fill_(attention[:, :width] == 0, 0)
    weight = model.lm_head.weight[offset:offset+size]
    bias = getattr(model.lm_head, "bias", None)
    if bias is not None:
        bias = bias[offset:offset+size]
    cache_args = {}
    if decode_backend == "cudagraph":
        from transformers.cache_utils import StaticCache
        cache_args = dict(past_key_values=StaticCache(config=model.config, max_cache_len=width+steps),
                          cache_position=torch.arange(width, device=device))
        compact_interval = 0  # Fixed graph/cache batch shape; outputs still have individual stop lengths.
    output = model.model(input_ids=ids, attention_mask=attention[:, :width],
                         position_ids=positions, use_cache=True, **cache_args)
    cache = output.past_key_values
    logits = F.linear(output.last_hidden_state[:, -1], weight, bias).float()
    graph_decode = None
    if decode_backend == "cudagraph":
        graph_decode = _CudagraphSpeechDecoder(model, cache, weight, bias,
                                               attention, lengths, width)
    minimum = torch.tensor(minimum_tokens, device=device)
    maximum = torch.tensor(maximum_tokens, device=device)
    generated = torch.zeros((batch_size, steps), dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    stop_steps = torch.full((batch_size,), steps, dtype=torch.long, device=device)
    active = torch.arange(batch_size, device=device)
    k = min(top_k, size)
    for step in range(steps):
        scores = logits / temperature
        penalized = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
        scores = torch.where(seen[active], penalized, scores)
        scores = torch.round(scores / logit_quantization) * logit_quantization
        scores[:, base:] = scores[:, base:].masked_fill((step < minimum[active])[:, None], -torch.inf)
        values, indices = torch.topk(scores, k=k, dim=-1)
        probabilities = torch.softmax(values.double(), dim=-1)
        remove = probabilities.cumsum(-1) - probabilities > top_p
        probabilities.masked_fill_(remove, 0)
        probabilities /= probabilities.sum(-1, keepdim=True).clamp_min(1e-12)
        chosen = (probabilities.cumsum(-1) < uniforms[active, step:step+1]).sum(-1).clamp(max=k-1)
        token = indices.gather(1, chosen[:, None]).squeeze(1)
        special = token >= base
        forced = step + 1 >= maximum[active]
        was_finished = finished[active]
        stop_steps[active] = torch.where((~was_finished) & special, step, stop_steps[active])
        stop_steps[active] = torch.where((~was_finished) & (~special) & forced, step+1, stop_steps[active])
        generated[active, step] = torch.where(was_finished | special, 0, token)
        global_token = torch.where(was_finished, eos_id, token + offset)
        seen[active, token] = True
        finished[active] |= special | forced
        if bool(finished.all()):
            break
        attention[active, width+step] = (~finished[active]).long()
        if compact_interval > 0 and (step+1) % compact_interval == 0:
            keep = (~finished[active]).nonzero(as_tuple=True)[0]
            if len(keep) < len(active):
                cache.batch_select_indices(keep)
                active = active[keep]
                global_token = global_token[keep]
        if graph_decode is not None:
            logits = graph_decode(global_token, attention, lengths+step, width+step)
        else:
            output = model.model(input_ids=global_token[:, None],
                                 attention_mask=attention[active, :width+step+1],
                                 position_ids=(lengths[active] + step)[:, None],
                                 past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            logits = F.linear(output.last_hidden_state[:, -1], weight, bias).float()
    values = generated.cpu()
    return [values[row, :n].tolist() for row, n in enumerate(stop_steps.cpu().tolist())]


class _CudagraphSpeechDecoder:
    """One fixed-shape decode graph per batch; explicit mask excludes future KV.

    Warmup writes only the first decode cache slot, which replay overwrites with
    the real sampled token before reading it. All later slots stay masked until
    their actual token has been written. No global model monkey-patches are used.
    """
    def __init__(self, model, cache, weight, bias, attention, lengths, width):
        self.token = torch.zeros((len(lengths), 1), dtype=torch.long, device=lengths.device)
        self.positions = lengths[:, None].clone()
        self.cache_position = torch.tensor([width], device=lengths.device)
        self.mask = attention[:, None, None, :].bool().clone()
        self.mask[:, :, :, width] = True
        self.cache = cache
        def forward():
            hidden = model.model(input_ids=self.token,
                                 attention_mask={"full_attention": self.mask},
                                 position_ids=self.positions, past_key_values=self.cache,
                                 cache_position=self.cache_position, use_cache=True).last_hidden_state[:, -1]
            return F.linear(hidden, weight, bias).float()
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(warmup)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = forward()

    def __call__(self, token, attention, positions, cache_position):
        self.token.copy_(token[:, None])
        self.positions.copy_(positions[:, None])
        self.cache_position.fill_(cache_position)
        self.mask.copy_(attention[:, None, None, :].bool())
        self.graph.replay()
        return self.logits


def plan_cosyvoice3_batches(prepared, *, batch_size: int, mode: str = "bucket",
                           input_token_budget: int = 32768):
    """Group nearby prompt lengths, then nearby target lengths within windows.

    The token budget limits padded prefill size, not text or audio length.
    Entries are (record, input_ids, target_text_token_count).
    """
    if batch_size < 1 or input_token_budget < 1 or mode not in {"bucket", "exact"}:
        raise ValueError("invalid batch planner configuration")
    ordered = sorted(prepared, key=lambda x: (len(x[1]), x[2], str(x[0]["utterance_id"])))
    if mode == "bucket":
        ordered = [x for start in range(0, len(ordered), batch_size * 4)
                   for x in sorted(ordered[start:start+batch_size*4], key=lambda x: (x[2], len(x[1])))]
    batches, current, maximum = [], [], 0
    for item in ordered:
        length = len(item[1])
        if current and (len(current) >= batch_size
                        or max(maximum, length) * (len(current)+1) > input_token_budget
                        or (mode == "exact" and length != maximum)):
            batches.append(current)
            current, maximum = [], 0
        current.append(item)
        maximum = max(maximum, length)
    if current:
        batches.append(current)
    return batches


def plan_token2wav_batches(lengths, *, batch_size: int, token_budget: int):
    """Pack true prompt+target lengths; one outlier never shrinks every batch."""
    if batch_size < 1 or token_budget < 1 or any(n < 1 for n in lengths):
        raise ValueError("invalid acoustic batch configuration")
    batches, current = [], []
    for i in sorted(range(len(lengths)), key=lambda i: lengths[i]):
        if current and (len(current) >= batch_size or lengths[i]*(len(current)+1) > token_budget):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


def _length_mask(lengths: Tensor, maximum: int) -> Tensor:
    positions = torch.arange(maximum, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


@torch.inference_mode()
def _causal_cfm_batch(
    decoder: torch.nn.Module,
    *,
    mu: Tensor,
    mask: Tensor,
    speaker_conditions: Tensor,
    conditions: Tensor,
    diffusion_steps: int,
    streaming: bool,
    estimator_precision: str = "fp32",
) -> Tensor:
    """Vectorize upstream CausalConditionalCFM's hard-coded batch-one solver."""

    if not isinstance(decoder.estimator, torch.nn.Module):
        raise TypeError("true batching currently requires the PyTorch DiT estimator")
    batch_size = int(mu.shape[0])
    # Upstream deliberately reuses the same fixed noise for deterministic
    # inference. Expanding it preserves each sample's batch-one trajectory.
    state = decoder.rand_noise[:, :, : mu.shape[2]].to(mu).expand(
        batch_size, -1, -1
    ).clone()
    timeline = torch.linspace(
        0, 1, diffusion_steps + 1, device=mu.device, dtype=mu.dtype
    )
    if decoder.t_scheduler == "cosine":
        timeline = 1 - torch.cos(timeline * 0.5 * torch.pi)
    guidance = float(decoder.inference_cfg_rate)
    current = timeline[0]
    for step in range(1, len(timeline)):
        delta = timeline[step] - current
        doubled_state = torch.cat((state, state), dim=0)
        doubled_mask = torch.cat((mask, mask), dim=0)
        doubled_mu = torch.cat((mu, torch.zeros_like(mu)), dim=0)
        doubled_time = current.expand(2 * batch_size)
        doubled_speakers = torch.cat(
            (speaker_conditions, torch.zeros_like(speaker_conditions)), dim=0
        )
        doubled_conditions = torch.cat(
            (conditions, torch.zeros_like(conditions)), dim=0
        )
        # Keep Euler integration and the fixed initial noise in FP32. Only the
        # expensive DiT forward optionally uses BF16 Tensor Core arithmetic.
        with torch.autocast(device_type=mu.device.type, dtype=torch.bfloat16,
                            enabled=estimator_precision == "bf16"):
            derivative = decoder.estimator(
                doubled_state, doubled_mask, doubled_mu, doubled_time,
                doubled_speakers, doubled_conditions, streaming=streaming,
            )
        derivative = derivative.float()
        conditional, unconditional = derivative.chunk(2, dim=0)
        derivative = (1.0 + guidance) * conditional - guidance * unconditional
        state = state + delta * derivative
        current = timeline[step]
    return state.float()


@torch.inference_mode()
def flow_inference_batch(
    flow: torch.nn.Module,
    *,
    generated_tokens: Sequence[Sequence[int]],
    prompt_tokens: Sequence[Sequence[int]],
    prompt_features: Sequence[Tensor],
    speaker_embeddings: Tensor,
    streaming: bool = False,
    finalize: bool = True,
    diffusion_steps: int = 10,
    estimator_precision: str = "fp32",
) -> tuple[Tensor, ...]:
    """Run a true padded CosyVoice3 flow batch and return unpadded mels."""

    batch_size = len(generated_tokens)
    if estimator_precision not in {"fp32", "bf16"}:
        raise ValueError("estimator_precision must be fp32 or bf16")
    if batch_size < 1:
        return ()
    if not (
        len(prompt_tokens) == batch_size
        and len(prompt_features) == batch_size
        and tuple(speaker_embeddings.shape)
        == (batch_size, COSYVOICE3_SPEAKER_DIMENSION)
    ):
        raise ValueError("CosyVoice3 batch fields have incompatible sizes")
    if not finalize:
        raise NotImplementedError("padded batch flow currently requires finalize=True")

    device = next(flow.parameters()).device
    token_items: list[Tensor] = []
    total_token_lengths: list[int] = []
    prompt_mel_lengths: list[int] = []
    target_mel_lengths: list[int] = []
    aligned_prompt_features: list[Tensor] = []
    ratio = int(flow.token_mel_ratio)
    for target, prompt, feature in zip(
        generated_tokens, prompt_tokens, prompt_features, strict=True
    ):
        if not target:
            raise ValueError("generated speech-token sequence must be non-empty")
        if feature.ndim != 2 or feature.shape[1] != int(flow.output_size):
            raise ValueError("prompt feature must have shape (frames, flow_output_size)")
        if prompt:
            aligned_prompt_length = min(len(prompt), int(feature.shape[0]) // ratio)
            if aligned_prompt_length < 1:
                raise ValueError("prompt is too short after token/mel alignment")
        else:
            if int(feature.shape[0]) != 0:
                raise ValueError(
                    "prompt feature must be empty when prompt speech tokens are empty"
                )
            aligned_prompt_length = 0
        prompt_tensor = torch.as_tensor(
            prompt[:aligned_prompt_length], dtype=torch.long
        )
        target_tensor = torch.as_tensor(target, dtype=torch.long)
        token_items.append(torch.cat((prompt_tensor, target_tensor)))
        total_token_lengths.append(len(prompt_tensor) + len(target_tensor))
        prompt_mel_lengths.append(len(prompt_tensor) * ratio)
        target_mel_lengths.append(len(target_tensor) * ratio)
        aligned_prompt_features.append(feature[: len(prompt_tensor) * ratio].float())

    token_lengths = torch.tensor(total_token_lengths, dtype=torch.long, device=device)
    token = pad_sequence(token_items, batch_first=True, padding_value=0).to(device)
    token_mask = _length_mask(token_lengths, token.shape[1]).unsqueeze(-1)

    embedding = F.normalize(speaker_embeddings.to(device=device, dtype=torch.float32), dim=1)
    embedding = flow.spk_embed_affine_layer(embedding)
    token_hidden = flow.input_embedding(torch.clamp(token, min=0)) * token_mask.to(
        dtype=embedding.dtype
    )
    hidden = flow.pre_lookahead_layer(token_hidden)
    hidden = hidden.repeat_interleave(ratio, dim=1)

    mel_lengths = token_lengths * ratio
    maximum_mel = int(hidden.shape[1])
    conditions = hidden.new_zeros((batch_size, maximum_mel, int(flow.output_size)))
    for index, feature in enumerate(aligned_prompt_features):
        feature = feature.to(device=device, dtype=hidden.dtype)
        conditions[index, : feature.shape[0]] = feature
    mel_mask = _length_mask(mel_lengths, maximum_mel).unsqueeze(1).to(hidden)

    feature = _causal_cfm_batch(
        flow.decoder,
        mu=hidden.transpose(1, 2).contiguous(),
        mask=mel_mask,
        speaker_conditions=embedding,
        conditions=conditions.transpose(1, 2).contiguous(),
        diffusion_steps=diffusion_steps,
        streaming=streaming,
        estimator_precision=estimator_precision,
    )
    outputs: list[Tensor] = []
    for index, (prompt_length, target_length) in enumerate(
        zip(prompt_mel_lengths, target_mel_lengths, strict=True)
    ):
        outputs.append(feature[index, :, prompt_length : prompt_length + target_length].float())
    return tuple(outputs)


@torch.inference_mode()
def hift_inference_batch(
    hift: torch.nn.Module,
    mels: Sequence[Tensor],
    *,
    samples_per_mel: int = 480,
) -> tuple[Tensor, ...]:
    """Run causal HiFT on one padded batch and crop valid waveform prefixes."""

    if not mels:
        return ()
    if samples_per_mel < 1:
        raise ValueError("samples_per_mel must be positive")
    channels = {int(mel.shape[0]) for mel in mels if mel.ndim == 2}
    if channels != {80} or any(mel.ndim != 2 for mel in mels):
        raise ValueError("every mel must have shape (80, frames)")
    device = next(hift.parameters()).device
    lengths = [int(mel.shape[1]) for mel in mels]
    padded = pad_sequence(
        [mel.transpose(0, 1) for mel in mels],
        batch_first=True,
        padding_value=0.0,
    ).transpose(1, 2).to(device)
    waveforms, _ = hift.inference(speech_feat=padded, finalize=True)
    return tuple(
        waveforms[index : index + 1, : length * samples_per_mel].float()
        for index, length in enumerate(lengths)
    )


def compare_single_and_batch_audio(
    singles: Sequence[Tensor],
    batched: Sequence[Tensor],
) -> dict[str, float]:
    """Return strict shape and numerical diagnostics for the batch gate."""

    if len(singles) != len(batched):
        raise ValueError("single and batch result counts differ")
    maximum = 0.0
    mean = 0.0
    samples = 0
    for single, batch in zip(singles, batched, strict=True):
        if single.shape != batch.shape:
            raise ValueError(
                f"single/batch waveform shapes differ: {single.shape} vs {batch.shape}"
            )
        difference = (single.float().cpu() - batch.float().cpu()).abs()
        maximum = max(maximum, float(difference.max()))
        mean += float(difference.sum())
        samples += difference.numel()
    return {
        "waveform_max_abs": maximum,
        "waveform_mean_abs": mean / max(samples, 1),
        "compared_samples": float(samples),
    }
