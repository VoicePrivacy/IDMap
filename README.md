# IDMap: A Pseudo-Speaker Generator Framework Based on Speaker Identity Index to Vector Mapping
## Demo Page
# You can access the audio samples at  [https://voiceprivacy.github.io/IDMap/).
## Project Overview
This project implements a speech anonymization method based on IDMap. The method combines acoustic features and emotional information to protect speech privacy while preserving emotional features. 

## File Structure
- **infer.py**: This file contains the inference code for the model.
- **models_U2SAnon.py**: This file defines the model architecture, including the network structure used for generating UIDVs and for speech anonymization.

## Content Embedding Extraction
The extraction of content embeddings can be achieved using the code from the repository [https://github.com/deep-privacy/SA-toolkit/tree/master/egs/asr/librispeech).
We will provide the pre-trained model weights used in paper. You can download them from the link at the end of this document.

## Speaker Embedding Extraction
In this project, you can extract speaker embeddings using your own dataset. The steps are as follows:

1. **Extract Speaker Embeddings from Your Own Dataset**:
   - You can follow the process described in the paper to extract speaker embeddings for speaker information.

2. **Verify the Effectiveness**:
   - During training, use the extracted speaker embeddings to evaluate the anonymization effect.
   - Compare the results using different speaker embeddings extracted from various datasets.

Or you can download the Speaker embedding compressed package provided by us for experiment. The download link is provided at the end of the documentation.

### IDV Generation Process
1. The IDV is a 512-dimensional vector
2. You can use the functions of 'numpy.random.normal' or 'numpy.random.uniform' in numpy to generate corresponding identity features. Note that each identity feature needs to be set with a unique random seed. egs:
```python
import numpy as np

def generate_uniform_idv(seed, dim=512):
    np.random.seed(seed)  
    return np.random.uniform(-1, 1, dim) 

if __name__ == "__main__":
    num_speakers = 10
    for speaker_id in range(num_speakers):
        idv = generate_uniform_idv(seed=speaker_id)
```


