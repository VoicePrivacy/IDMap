Acoustic model / ASR
===

This folder contains librispeech recipes to train ASR / linguistic feature extractor.

To run the recipe:

```bash
# Activate your miniconda env and kaldi env
. ./path.sh

#  Change the path to librispeech database in `configs/local.conf` and/or use `local/download_libri.sh`
./local/chain/prepare_data.sh --train_set train_clean_100
./local/chain/prepare_data.sh --train_set train_clean_360
./local/chain/prepare_data.sh --train_set train_600 # train-clean-100 + train-other-500

# Train with archi and data defined in configs and local/chain/tuning/ (configs: model_file)
local/chain/train.py --conf configs/...
```

_Up to 5 gpus can be used for training, you can use `ssh.pl` to distribute training on multiple node or max_concurrent_jobs config[exp] option to sequence the training, (using natural gradient and parameter averaging)._

### Content extraction

The script for content feature extraction is provided in extract_asrbn.py.
