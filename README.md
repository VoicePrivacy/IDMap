# IDMap: backend-native anonymous speaker vectors

IDMap maps a reproducible anonymous identity index to a speaker vector in the
**specific speaker-conditioning space used by a synthesizer**. This repository
retains the original paper code in [`IDMap-MLP/`](IDMap-MLP/) and
[`IDMap-Diff/`](IDMap-Diff/). The maintained backend-native training and
synthesis implementation is in [`src/voice_anon/`](src/voice_anon/) and
[`scripts/`](scripts/). The two implementations and their checkpoints are not
interchangeable.

The original paper is *Improving the Uniqueness and Efficiency in Voice
Anonymization with Index to Vector Mapping*. Its [audio samples](https://voiceprivacy.github.io/IDMap/)
and [legacy usage notes](docs/legacy_idmap.md) remain available.

## What is currently released

| Generator | IDMap-MLP speaker space | Verified inference weight | IDMap-Diffusion weight |
| --- | --- | --- | --- |
| Qwen3-TTS 12Hz 0.6B Base | Native 1024-D x-vector | [Download](https://github.com/VoicePrivacy/IDMap/releases/download/v0.1.0-native-idmap/qwen3tts_librispeech360_1024d_idmap_mlp_inference.pt) | Not verified or released |
| Original CosyVoice3 0.5B | Native 192-D CAM++ | [Download](https://github.com/VoicePrivacy/IDMap/releases/download/v0.1.0-native-idmap/cosyvoice3_campplus_192d_idmap_mlp_inference.pt) | Not verified or released |

See [checkpoint checksums and provenance](checkpoints/README.md). These are
inference-only IDMap weights, **not** the Qwen3-TTS or CosyVoice3 generator
weights. Download the latter from their official publishers. In particular,
do not feed a 192-D CosyVoice3 IDMap vector to Qwen3-TTS or vice versa.
The Qwen3-TTS MLP completed ten LibriSpeech train-clean-360 epochs; its small
development rendering check is **not** a formal VPC result. No corresponding
IDMap-Diffusion checkpoint has passed provenance and compatibility checks, so
none is linked here.

## Install

Use separate environments for Qwen3-TTS and CosyVoice3, because their vendor
dependencies differ. A CUDA-compatible PyTorch and torchaudio installation is
required for synthesis; install those from the PyTorch selector for your CUDA
runtime before installing this package. Python 3.11 is recommended for the
maintained code; the legacy code has separate requirements. Verify that
PyTorch wheels exist for your OS and Python version before setup.

```bash
git clone https://github.com/VoicePrivacy/IDMap.git
cd IDMap
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[audio,test]'
python -m pytest -q tests
```

For Qwen3-TTS, additionally install the official `qwen-tts` runtime and
download [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).
For CosyVoice3, follow the [official CosyVoice installation](https://github.com/FunAudioLLM/CosyVoice),
including its Matcha-TTS and S3Tokenizer dependencies, and download
[Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512).
The CosyVoice3 batch worker additionally needs a `hf_merged/` directory
produced by the vendor `runtime/triton_trtllm/scripts/convert_cosyvoice3_to_hf.py`
script. Downloading the base checkpoint alone is insufficient. Run the worker
with `PYTHONPATH` including the CosyVoice vendor packages and Matcha-TTS.
See the [detailed environment and conversion guide](docs/native_training_and_synthesis.md).

## Download and check IDMap weights

```bash
mkdir -p checkpoints/downloaded
curl -fL -o checkpoints/downloaded/qwen-idmap-mlp.pt \
  https://github.com/VoicePrivacy/IDMap/releases/download/v0.1.0-native-idmap/qwen3tts_librispeech360_1024d_idmap_mlp_inference.pt
curl -fL -o checkpoints/downloaded/cosy-idmap-mlp.pt \
  https://github.com/VoicePrivacy/IDMap/releases/download/v0.1.0-native-idmap/cosyvoice3_campplus_192d_idmap_mlp_inference.pt
sha256sum checkpoints/downloaded/*.pt  # macOS: shasum -a 256
```

Compare both hashes with [the checksum table](checkpoints/README.md) before
loading. Only load trusted PyTorch checkpoints.

For a structural CPU check of the Qwen export (this does not validate audio
quality), run:

```bash
python scripts/verify_idmap_checkpoint.py \
  --checkpoint checkpoints/downloaded/qwen-idmap-mlp.pt \
  --expected-sha256 0b723695b6c21d748e6fc583cbadeadad84eec7429a11beb8af7d5f63c92dabb \
  --expected-dimension 1024 \
  --expected-speaker-space qwen3tts-12hz-0p6b-base-xvector-v1:87114bb1ca84f9f4
```

## Train an IDMap in a new generator's speaker space

1. Extract per-utterance **native** speaker embeddings from the exact vendor
   model you will synthesize with. The provided extractors are
   [`extract_qwen3tts_speaker_embeddings.py`](scripts/extract_qwen3tts_speaker_embeddings.py)
   and [`extract_cosyvoice3_speaker_embeddings.py`](scripts/extract_cosyvoice3_speaker_embeddings.py).
   Organize the input as `AUDIO_ROOT/SPEAKER_ID/.../recording.wav` or `.flac`.
2. Save an `.npz` with float32 `embeddings[N,D]` and string `speaker_ids[N]`.
   The extractors record the encoder fingerprint and audit batch/single output.
   Speaker labels are training-only; never feed them to the inference worker.
3. Train MLP or Diffusion with the **matching** dimension and speaker-space
   label. The Qwen weight above used the additional
   [native-diversity training recipe](scripts/train_idmap_native_diversity_full_20260904.py),
   not the generic MLP command below.

```bash
python scripts/train_idmap_mlp.py \
  --embeddings /path/to/native_embeddings.npz \
  --speaker-space 'cosyvoice3-campplus-v1:<encoder-hash-prefix>' \
  --output-dir /path/to/mlp-run
python scripts/train_idmap_diff.py \
  --embeddings /path/to/native_embeddings.npz \
  --speaker-space 'cosyvoice3-campplus-v1:<encoder-hash-prefix>' \
  --variant paper_vp_sde --output-dir /path/to/diff-run
python scripts/export_idmap_inference_checkpoint.py \
  --checkpoint /path/to/mlp-run/best.pt --output /path/to/mlp-inference.pt
```

Do not treat a finite training loss or a saved Diffusion checkpoint as proof of
anonymous voice quality. Validate generated-vector distribution, identity
separation, synthesized audio, WER, and privacy with the matching ASV protocol.

## Synthesize anonymized audio

Create one JSON object per line in `manifest.jsonl`:

```json
{"utterance_id":"session1-turn1","text":"Hello there.","anonymous_index":12345,"output_relative_path":"session1/turn1.wav"}
{"utterance_id":"session1-turn2","text":"I agree.","anonymous_index":12345,"output_relative_path":"session1/turn2.wav"}
```

The repeated index intentionally preserves the same pseudo-speaker across
turns. Different source identities need different indices, assigned by your
own diarization/tracker. IDMap itself does **not** infer source identity or
transcribe speech. The worker synthesizes the supplied text; it does not read
source audio. For VPC-style per-utterance re-randomization, assign a different
index to each utterance. The Qwen worker writes 16-kHz PCM WAVs.

For a large manifest, [`prepare_idmap_manifest.py`](scripts/prepare_idmap_manifest.py)
assigns distinct indices without collisions. Its input JSONL needs
`utterance_id` and `text`; `--mode utterance` gives every utterance a new
pseudo-speaker. With `--mode session`, each row additionally needs `session_id`
and `local_speaker_id` from **your own evaluated speaker tracker**. It reuses
one index for repeated local IDs and rotates it between sessions. It does not
derive speaker IDs from the audio or permit oracle labels in a claimed
end-to-end evaluation.

```bash
python scripts/prepare_idmap_manifest.py \
  --input-jsonl /path/to/text_and_tracker_output.jsonl \
  --output manifest.jsonl --mode session --seed 20260924
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_qwen3tts_idmap_worker.py \
  --manifest manifest.jsonl --output-dir /path/to/anonymized \
  --model-dir /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --idmap-checkpoint checkpoints/downloaded/qwen-idmap-mlp.pt \
  --expected-speaker-space-prefix qwen3tts-12hz-0p6b-base-xvector-v1 \
  --rank 0 --world-size 1 --batch-size 16
```

For CosyVoice3, use `generate_cosyvoice3_multigpu_worker.py` with the 192-D
checkpoint, `--prompt-mode none`, the original vendor checkpoint and the
converted `hf_merged` directory. Exact commands, batching caveats, and
checkpoint-compatibility checks are in the
[training and synthesis guide](docs/native_training_and_synthesis.md).
The [launcher](scripts/launch_idmap_generation.py) starts one worker per GPU,
sets rank/world-size consistently, and writes an audit with the expected,
valid and failed WAV denominators. For example:

```bash
python scripts/launch_idmap_generation.py --backend qwen --gpus 0,1 \
  --manifest manifest.jsonl --output-dir /path/to/qwen-audio \
  --model-dir /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --idmap-checkpoint checkpoints/downloaded/qwen-idmap-mlp.pt \
  --expected-speaker-space-prefix qwen3tts-12hz-0p6b-base-xvector-v1 \
  --batch-size 16

python scripts/launch_idmap_generation.py --backend cosy --gpus 2,3 \
  --manifest manifest.jsonl --output-dir /path/to/cosy-audio \
  --model-dir /path/to/Fun-CosyVoice3-0.5B-2512 \
  --hf-model-dir /path/to/Fun-CosyVoice3-0.5B-2512/hf_merged \
  --idmap-checkpoint checkpoints/downloaded/cosy-idmap-mlp.pt \
  --batch-size 8
```

The selected GPUs must actually be free; the launcher never stops other
processes. Audit the output WAV count and ASR quality before reporting a
result. Qwen supports batched text, but batch/single waveforms are not claimed
identical under stochastic decoding. The two backends use separate native
speaker spaces and separate model weights.

## Legacy paper implementation

The original `IDMap-MLP/`, `IDMap-Diff/`, `SA-toolkit/`, examples and figures
remain for historical reproduction. They are not the backend-native code above;
install their separate [`requirements.txt`](requirements.txt) only in an
isolated legacy environment. Original paper figures: [EER/WER/UAR](figures/EER_WER_UAR.png),
[Gvd](figures/Gvd.png), [RTF](figures/RTFs.png).
