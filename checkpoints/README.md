# Backend-native checkpoint downloads

The Git repository contains code, not large model files. Verified inference-only
IDMap-MLP assets are attached to the [research pre-release](https://github.com/VoicePrivacy/IDMap/releases/tag/v0.1.0-native-idmap).

| Backend speaker space | File | SHA-256 |
| --- | --- | --- |
| Qwen3-TTS Base 1024-D x-vector | [qwen3tts_librispeech360_1024d_idmap_mlp_inference.pt](https://github.com/VoicePrivacy/IDMap/releases/download/v0.1.0-native-idmap/qwen3tts_librispeech360_1024d_idmap_mlp_inference.pt) | `0b723695b6c21d748e6fc583cbadeadad84eec7429a11beb8af7d5f63c92dabb` |
| CosyVoice3 192-D CAM++ | [cosyvoice3_campplus_192d_idmap_mlp_inference.pt](https://github.com/VoicePrivacy/IDMap/releases/download/v0.1.0-native-idmap/cosyvoice3_campplus_192d_idmap_mlp_inference.pt) | `23f2ef9212eaab0aa966538a3b2077661256e8868aa6c081990db16e9315cdeb` |

Download with `curl -fL -o FILE URL`, then verify with `sha256sum FILE`
(or `shasum -a 256 FILE` on macOS). These vectors are not interchangeable.
Official Qwen3-TTS and CosyVoice3 generator weights must be downloaded from
their respective model publishers; they are not bundled here.

The original paper's `IDMap-Diff/` implementation and the newer
`src/voice_anon/idmap/diffusion.py` training implementation are available.
**No compatible, verified IDMap-Diffusion trained checkpoint is published.**
Historical runs referenced an OIT path that is currently inaccessible. Do not
substitute an MLP checkpoint or an unverified diffusion checkpoint. The
`scripts/train_idmap_diff.py` entry point can produce a new checkpoint after
extracting backend-native speaker vectors.
