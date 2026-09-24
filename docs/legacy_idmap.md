# Original IDMap paper implementation

The original paper's inference sources are in [`IDMap-MLP/`](../Original/IDMap-MLP/)
and [`IDMap-Diff/`](../Original/IDMap-Diff/). They were written for the original
content/speaker/emotion feature pipeline, not for the 1024-D Qwen3-TTS or
192-D CosyVoice3 native-vector generators. Do not substitute their checkpoints
or commands into the maintained `src/voice_anon/` workers.

The original environment uses [`Original/requirements.txt`](../Original/requirements.txt) and the
[`SA-toolkit/`](../Original/SA-toolkit/) tree. The entry points are
`Original/IDMap-MLP/infer.py` and `Original/IDMap-Diff/infer.py`; they require their original feature/model assets and
are not turnkey commands for the new backend-native checkpoints. The paper's
[audio demo](https://voiceprivacy.github.io/IDMap/) and figures in
[`figures/`](../figures/) remain available.

For current installation, backend-native embedding extraction, training,
checkpoint download and generation, use the [repository README](../README.md)
and [native training guide](native_training_and_synthesis.md).
