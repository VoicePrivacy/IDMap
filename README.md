# U2SAnon: A Uniform to Speaker Vector Anonymizer in Voice Anonymization
## Demo Page
### **You can access the demo page** at [https://voiceprivacy.github.io/U2SAon/](https://voiceprivacy.github.io/U2SAon/).
## Project Overview
This project implements a speech anonymization method based on Uniform Identity Vector (UIDV). The method combines acoustic features and emotional information to protect speech privacy while preserving emotional features. The model training process is accelerated by using pre-generated UIDVs for training.

## File Structure
- **train_US2Anon.py**: This file contains the training code for the model. It is responsible for training the UIDV generator and the anonymization model.
- **models_U2SAnon.py**: This file defines the model architecture, including the network structure used for generating UIDVs and for speech anonymization.

## Model Training
1. **Training Code**: The model training process is defined in the `train_US2Anon.py` file. You can run this file to train the model.
2. **Model Architecture**: The model architecture is defined in the `models_U2SAnon.py` file. It includes definitions of all the components used in the training process.

## UIDV Generation and Assignment
To speed up the training process, pre-generated UIDVs are provided. You can also generate your own UIDV storage files following the process described in the paper.

### UIDV Generation Process
1. The UIDV is a 512-dimensional vector, with each dimension independently distributed between -1 and 1.
2. To generate UIDVs, we initialize an empty list and generate UIDVs by rotating feature dimensions.
3. During training, each UIDV is matched with other UIDVs to anonymize speaker identity.

In the `train_US2Anon.py` file, you can choose whether to use pre-generated UIDVs or generate and use your own UIDV files.

## Speaker Embedding Extraction
In this project, you can extract speaker embeddings using your own dataset. The steps are as follows:

1. **Extract Speaker Embeddings from Your Own Dataset**:
   - You can follow the process described in the paper to extract speaker embeddings based on emotional features and speaker information.
   - If you are using the provided dataset for training, make sure you have extracted the corresponding speaker embeddings and loaded them into the training process.

2. **Verify the Effectiveness**:
   - During training, use the extracted speaker embeddings to evaluate the anonymization effect.
   - Compare the results using different speaker embeddings extracted from various datasets.

## Training Steps
### Install Dependencies
Before training, make sure to install the required dependencies:
```bash
pip install -r requirements.txt
