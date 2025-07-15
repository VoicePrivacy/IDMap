# IDMap: A Pseudo-Speaker Generator Framework Based on Speaker Identity Index to Vector Mapping
## Demo Page
# You can access the audio samples at  [https://voiceprivacy.github.io/IDMap/].
## Project Overview
This project implements a speech anonymization method based on IDMap. The method combines acoustic features and emotional information to protect speech privacy while preserving emotional features. For the sake of AI privacy, we will not provide training scripts, but only the inference scripts.

### Install Dependencies
Before inference, make sure to install the required dependencies. First, create a virtual environment using conda:
```bash
conda create -n idmap python=3.9
conda activate idmap
```
Then proceed with the following steps:
```bash
git clone https://github.com/VoicePrivacy/IDMap.git
cd IDMap
pip install -r requirements.txt
```
The inference process can be completed after the virtual environment is installed.

### Inference Preparation
Our model requires three parts: content embedding, speaker embedding, and emotion features. 

The content embedding is extracted using the scripts provided at [https://github.com/deep-privacy/SA-toolkit/tree/master/egs/asr/librispeech](https://github.com/deep-privacy/SA-toolkit/tree/master/egs/asr/librispeech). The pre-trained models can be obtained from the download links at the bottom of the document. Additionally, we will provide the content embeddings used in the VPC 2024 evaluation for readers to download and use for evaluation.

The speaker embeddings used in training are extracted using the scripts provided at [https://github.com/Snowdar/asv-subtools/blob/master/pytorch/launcher/runEcapaXvector_online.py](https://github.com/Snowdar/asv-subtools/blob/master/pytorch/launcher/runEcapaXvector_online.py). In this project, they can be directly generated using the IDMap framework without extraction.

The emotion labels are extracted using the SER model and scripts provided at [https://github.com/Sreyan88/MMER](https://github.com/Sreyan88/MMER). We will provide the already-extracted labels for download. 

#### Preparation step
The pre-training model, as well as the characteristics of the corresponding data set used in our evaluation during the experiment, will be uploaded after finishing.

### Inference step
#### IDMap-MLP inference
After preparing the data, run `infer.py` to generate anonymized audio examples:

```bash
cd IDMap-MLP
python infer.py
```
In the future, we will provide code for anonymizing LibriSpeech 360, dev, test and other data sets.

#### IDMap-Diff inference
After preparing the data, run `infer.py` to generate anonymized audio examples:

```bash
cd IDMap-Diff
python infer.py
```
In the future, we will provide code for anonymizing LibriSpeech 360, dev, test and other data sets.

