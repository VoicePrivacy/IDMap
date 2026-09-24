# Backend-native IDMap training and synthesis

This directory supplements the original IDMap inference release. It contains the research implementation used to train separate IDMap models for each speech generator. **No speaker-embedding dataset or trained IDMap checkpoint is included in the Git repository.** These files must not be confused with the original paper's published weights.

## Layout

| Path | Purpose |
| --- | --- |
| `src/voice_anon/idmap/mlp.py` | MLP, deterministic index vectors, and cosine/Euclidean training loss |
| `src/voice_anon/idmap/diffusion.py` | VP-SDE paper variant and an additional EDM variant |
| `scripts/extract_qwen3tts_speaker_embeddings.py` | Native Qwen3-TTS Base 1024-D embedding extraction and batch/single audit |
| `scripts/extract_cosyvoice3_speaker_embeddings.py` | Native CosyVoice3 CAM++ 192-D extraction and batch/single audit |
| `scripts/train_idmap_mlp.py` | MLP training and resumable checkpoints |
| `scripts/train_idmap_native_diversity_full_20260904.py` | Exact full-corpus native-diversity variant used for the released Qwen3-TTS 1024-D checkpoint |
| `scripts/train_idmap_diff.py` | Diffusion training and resumable checkpoints |
| `scripts/verify_idmap_checkpoint.py` | Check trusted export SHA, speaker space, strict loading, and finite generated vectors |
| `scripts/prepare_idmap_manifest.py` | Convert text and externally assigned local speaker IDs to collision-free pseudo-speaker indices |
| `scripts/launch_idmap_generation.py` | Spawn one worker per selected GPU and audit all expected WAVs |
| `scripts/generate_qwen3tts_idmap_worker.py` | Batched Qwen3-TTS synthesis with native 1024-D IDMap-MLP or IDMap-Diff |
| `scripts/generate_cosyvoice3_multigpu_worker.py` | Batched CosyVoice3 synthesis with native 192-D IDMap-MLP or IDMap-Diff |

Install the backend's upstream runtime separately. Use a Python 3.10+ environment
and a CUDA-compatible PyTorch/torchaudio build for your machine. Then install
this package in editable mode with `python -m pip install -e '.[audio,test]'`.
Qwen synthesis also needs the official `qwen-tts` runtime (`python -m pip
install qwen-tts`). CosyVoice synthesis additionally needs the official
[CosyVoice source tree](https://github.com/FunAudioLLM/CosyVoice), Matcha-TTS,
S3Tokenizer, HyperPyYAML, Transformers, torchaudio, and soundfile. Install
CosyVoice using its own pinned vendor environment; add its `python_packages`,
repository root and `third_party/Matcha-TTS` paths to `PYTHONPATH`. The vendor
`runtime/triton_trtllm/scripts/convert_cosyvoice3_to_hf.py` must be run with
`--model-dir /path/to/Fun-CosyVoice3-0.5B-2512 --output-dir
/path/to/Fun-CosyVoice3-0.5B-2512/hf_merged --dtype bfloat16` before the
CosyVoice batch worker. The conversion needs the *original vendor* model
files; its output is not supplied by the official model download. Use
`python -m pytest -q tests` as a code smoke, not an audio-quality result.
The legacy top-level `requirements.txt` is for the original release and is
not a pinned native-backend environment.

## Data and training

Create one `.npz` archive containing `embeddings` (float32 array of shape
`[utterances, D]`), `speaker_ids` (same number of string labels), and
`speaker_space` (a scalar string fingerprint). The trainers now reject a
missing or mismatched `speaker_space`. Extract embeddings with the **same
encoder weights** as the target synthesizer: native Qwen3-TTS Base speaker
encoder (`D=1024`) or the exact CosyVoice3 `campplus.onnx` (`D=192`). The
extractors expect `AUDIO_ROOT/SPEAKER_ID/.../*.wav` or `.flac`; they use a
deterministic 3-second crop/repeat and check a small batch/single sample.
They do not extract dialogue identity labels. For a new evaluation, keep
training and test speakers disjoint. The training archives are not distributed
here.

```bash
# Qwen3-TTS extraction: one GPU, batched. Its JSON sidecar records speaker_space.
python scripts/extract_qwen3tts_speaker_embeddings.py \
  --audio-root /path/to/LibriSpeech/train-clean-360 \
  --model-dir /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --speaker-space-name qwen3tts-12hz-0p6b-base-xvector-v1 \
  --output /path/to/qwen-native-1024d.npz \
  --batch-size 128 --num-workers 8

# CosyVoice3 extraction: batched ONNX CAM++, no Qwen model involved.
python scripts/extract_cosyvoice3_speaker_embeddings.py \
  --audio-root /path/to/LibriSpeech/train-clean-360 \
  --campplus-onnx /path/to/Fun-CosyVoice3-0.5B-2512/campplus.onnx \
  --output /path/to/cosy-native-192d.npz \
  --batch-size 64 --num-workers 8
```

Read the exact `speaker_space` value from the extractor's `.manifest.json`
sidecar and pass it unchanged to `--speaker-space` below. Do not invent the
fingerprint or mix files from different vendor revisions.

```bash
export PYTHONPATH="$PWD/src"
python scripts/train_idmap_mlp.py \
  --embeddings /path/to/native_embeddings.npz \
  --speaker-space 'cosyvoice3-campplus-v1:<actual-campplus-sha256-prefix>' \
  --output-dir /path/to/idmap-mlp-run
python scripts/train_idmap_diff.py \
  --embeddings /path/to/native_embeddings.npz \
  --speaker-space 'cosyvoice3-campplus-v1:<actual-campplus-sha256-prefix>' \
  --output-dir /path/to/idmap-diff-run --variant paper_vp_sde
```

For Qwen, use the **actual extractor-recorded**
`qwen3tts-12hz-0p6b-base-xvector-v1:<model-fingerprint-prefix>` label and
1024-D embeddings. The `edm` diffusion variant is an additional
backend-normalized experiment, not the original paper variant. Every
`best.pt` contains weights, configuration, dimensionality, and a fixed
auxiliary vector. Only load checkpoints you trust: PyTorch pickle files can
execute code. A newly trained IDMap-Diff must be evaluated in the same
generator/encoder space before its weight is released.

Before publishing one of our own checkpoints, run `scripts/export_idmap_inference_checkpoint.py --checkpoint /path/to/best.pt --output /path/to/inference.pt`. The export removes optimizer/RNG state, local training paths, and the training-speaker list while preserving the fields required by both synthesis workers. Verify its printed SHA-256 and run a synthesis smoke against the intended vendor model before uploading.

## Synthesis manifest

One JSON object per line:

```json
{"utterance_id":"example-001","text":"Hello there.","anonymous_index":12345,"output_relative_path":"example-001.wav"}
```

Reuse one `anonymous_index` for every turn of the same session-local identity; use distinct indices for different identities. Do not use a reference transcript, RTTM, or a ground-truth speaker label as a prediction-time input. Manifest outputs must be relative paths. The workers shard records by `rank/world-size`; give each rank its own visible GPU.

Qwen3-TTS Base (vector-only, no reference audio; replace the checkpoint with a 1024-D IDMap-Diff checkpoint to use diffusion):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_qwen3tts_idmap_worker.py \
  --manifest /path/to/manifest.jsonl --output-dir /path/to/output \
  --model-dir /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --idmap-checkpoint /path/to/qwen-1024d-idmap-mlp/best.pt \
  --expected-speaker-space-prefix qwen3tts-12hz-0p6b-base-xvector-v1 \
  --rank 0 --world-size 1 --batch-size 16
```

CosyVoice3 (vector-only prompt mode; the model's native 192-D IDMap checkpoint is required):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_cosyvoice3_multigpu_worker.py \
  --manifest /path/to/manifest.jsonl --output-dir /path/to/output \
  --model-dir /path/to/Fun-CosyVoice3-0.5B \
  --hf-model-dir /path/to/Fun-CosyVoice3-0.5B/hf_merged \
  --idmap-checkpoint /path/to/cosy-192d-idmap-mlp/best.pt \
  --prompt-mode none --rank 0 --world-size 1 --llm-batch-size 8
```

The CosyVoice worker also accepts an IDMap-Diff checkpoint in the same native space. It expects the `hf_merged` layout and metadata generated by the upstream CosyVoice `runtime/triton_trtllm/scripts/convert_cosyvoice3_to_hf.py` conversion script, which is **not** an official downloaded model layout. Do not claim that downloading the upstream model alone makes that worker runnable. The worker verifies the checkpoint's `campplus.onnx` hash. Qwen verifies its checkpoint against the selected base-model fingerprint.

## Model downloads

- Official Qwen3-TTS Base weights: [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).
- Official CosyVoice3 weights: [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512).
- CosyVoice3 native 192-D IDMap-MLP inference checkpoint: [download from the v0.1.0 research pre-release](https://github.com/VoicePrivacy/IDMap/releases/download/v0.1.0-native-idmap/cosyvoice3_campplus_192d_idmap_mlp_inference.pt). SHA-256: `23f2ef9212eaab0aa966538a3b2077661256e8868aa6c081990db16e9315cdeb`. Speaker space: `cosyvoice3-campplus-v1:a6ac6a63997761ae`. The exported checkpoint was structurally loaded and generated finite native vectors; full vendor synthesis with this export has not yet been rerun.
- Qwen3-TTS native 1024-D IDMap-MLP inference checkpoint: [download from the v0.1.0 research pre-release](https://github.com/VoicePrivacy/IDMap/releases/download/v0.1.0-native-idmap/qwen3tts_librispeech360_1024d_idmap_mlp_inference.pt). SHA-256: `0b723695b6c21d748e6fc583cbadeadad84eec7429a11beb8af7d5f63c92dabb`. Speaker space: `qwen3tts-12hz-0p6b-base-xvector-v1:87114bb1ca84f9f4`. This LibriSpeech train-clean-360 run completed 10 epochs/4070 updates, with epoch 7/step 2849 selected by development distribution loss. A 32-item development rendering gate reported batch WER 1.969% and identity retrieval 30/32; it is not a formal VPC result. The exported checkpoint was structurally loaded and generated finite native vectors, but full vendor synthesis with this export has not yet been rerun.
- The released Qwen3-TTS checkpoint was produced by the **native-diversity variant**, not by the generic `train_idmap_mlp.py` recipe above. Its exact training loop is `scripts/train_idmap_native_diversity_full_20260904.py`, with loss implementation in `src/voice_anon/idmap/native_diversity.py`. It takes the native 1024-D embedding archive, a source-audit JSON, and an initial MLP checkpoint; those input artifacts are not distributed. Its development selection uses unseen identity indices but the same training speakers, not a speaker-disjoint development set.
- Backend-specific IDMap-Diff checkpoints: **not yet hosted**. The CosyVoice3 and Qwen3-TTS IDMap-MLP checkpoints are not interchangeable. No diffusion download link is claimed until trained bytes and compatibility are verified.

These upstream links are to the vendors' weights, not fine-tuned weights produced by this repository. Follow each upstream model's license and access terms.
