<div align="center">
<h1 align='center'>SA-toolkit</h1>
<img src="https://user-images.githubusercontent.com/7476655/232308795-90cef60d-08dd-4964-96cd-2afb4a6c03b0.jpg" width="25%">
<h2 align='center'>SA-toolkit: Speaker speech anonymization toolkit in python</h2>
</div>

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-lightgray)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deep-privacy/SA-toolkit/blob/master/SA-colab.ipynb)
[![Gradio demo](https://img.shields.io/website-up-down-yellow-red/https/hf.space/gradioiframe/Champion/SA-toolkit/+.svg?label=🤗%20Hugging%20Face-Spaces%20demo)](https://huggingface.co/spaces/Champion/SA-toolkit)

SA-toolkit is a [pytorch](https://pytorch.org/)-based library providing pipelines and basic building blocs designing speaker anonymization techniques.  
This library is the result of the work of Pierre Champion's [thesis](https://arxiv.org/abs/2308.04455).  

The extraction of content features is based on the ASR part in the SA-toolkit.

## Installation

### :snake: conda
The best way to install the SA-toolkit is with the `install.sh` script, which setup a micromamba environment, and kaldi.  
Take a look at the script and adapt it to your cluster configuration, or leave it do it's magic.  
This install is recommended for training ASR models.

```sh
git clone https://github.com/deep-privacy/SA-toolkit
./install.sh
```

### :package: pip
Another way of installing SA-toolkit is with pip3, this will setup everything for inference/testing.  
```sh
pip3 install 'git+https://github.com/deep-privacy/SA-toolkit.git@master#egg=satools&subdirectory=satools'
```

## Model training and content feature extraction

Checkout the READMEs of _[egs/asr/librispeech](egs/asr/librispeech)


## Citation

This library is the result of the work of Pierre Champion's thesis.  
If you found this library useful in academic research, please cite:

```bibtex
@phdthesis{champion2023,
    title={Anonymizing Speech: Evaluating and Designing Speaker Anonymization Techniques},
    author={Pierre Champion},
    year={2023},
    school={Université de Lorraine - INRIA Nancy},
    type={Thesis},
}
```

(Also consider starring the project on GitHub.)

## Acknowledgements
* Idiap' [pkwrap](https://github.com/idiap/pkwrap)
* Jik876's [HifiGAN](https://github.com/jik876/hifi-gan)
* A.Larcher's [Sidekit](https://git-lium.univ-lemans.fr/speaker/sidekit)
* Organazers of the [VoicePrivacy](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022) Challenge

## License
Most of the software is distributed under Apache 2.0 License (http://www.apache.org/licenses/LICENSE-2.0); the parts distributed under other licenses are indicated by a `LICENSE` file in related directories.
