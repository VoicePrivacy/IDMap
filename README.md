# IDMap: A Pseudo-Speaker Generator Framework Based on Speaker Identity Index to Vector Mapping
## Demo Page
# You can access the audio samples at  [https://voiceprivacy.github.io/IDMap/).
## Project Overview
This project implements a speech anonymization method based on IDMap. The method combines acoustic features and emotional information to protect speech privacy while preserving emotional features. 

## File Structure
- **infer.py**: This file contains the inference code for the model.
- **models_U2SAnon.py**: This file defines the model architecture, including the network structure used for generating UIDVs and for speech anonymization.

### IDV Generation Process
1. The IDV is a 512-dimensional vector
2. You can use the functions of 'numpy.random.normal' or 'numpy.random.uniform' in numpy to generate corresponding identity features. Note that each identity feature needs to be set with a unique random seed. egs:
```python
import numpy as np

def generate_uniform_idv(seed, dim=512):
    """生成基于均匀分布的唯一IDV，通过seed保证可复现性"""
    np.random.seed(seed)  # 设置唯一种子（如speaker_id、哈希值）
    return np.random.uniform(-1, 1, dim)  # 在[-1, 1)区间生成dim维向量

# 示例：生成10个唯一IDV（seed从0到9）
if __name__ == "__main__":
    num_speakers = 10
    for speaker_id in range(num_speakers):
        idv = generate_uniform_idv(seed=speaker_id)
