
![WhatsApp Image 2024-03-15 at 20 32 44](https://github.com/Asanta4/LLMs---Offensive-or-not-/assets/136238984/d35ee3ec-d43f-4ca0-b021-3592a9d90456)

# Toxic & Harmful Text Classifier

## Overview
This project leverages advanced Machine Learning and Deep Learning methods to classify text as offensive or non-offensive. We utilize traditional Neural Networks and cutting-edge Large Language Models (LLMs) like Mistral-7B and Llama-2-7B. The project aims to enhance the detection of hate speech and harmful content across various platforms, contributing to safer online environments.

## Project Details

### Introduction
The rapid expansion of online platforms has underscored the importance of moderating content to prevent the spread of hate speech and harassment. This project explores the efficacy of different models in classifying text, using a comparative approach between traditional neural networks and LLMs.

### Data Sets
We employed three major datasets:
- **Stormfront Dataset**: Contains hate speech related data.
- **Wiki Dataset**: Consists of toxic comments from Wikipedia.
- **Jigsaw Dataset**: Includes multilingual toxicity data.

### Technologies Used
- **Programming Language**: Python
- **Libraries**: Keras, Hugging Face Transformers, Pandas, NumPy
- **Models**: Keras Neural Network, Mistral-7B, Llama-2-7B
- **Tools**: Jupyter Notebook

### Model Development
1. **Keras Neural Network**:
   - Preprocessing: Tokenization, lemmatization, removal of stop words.
   - Architecture: Conv1D, MaxPooling1D, Dropout, Dense layers.
   - Performance measured by binary accuracy.
2. **Large Language Models (LLMs)**:
   - Mistral and Llama models were fine-tuned and evaluated using prompt engineering.
   - Techniques like LoRa (Low-Rank Adaptation) were used for efficient training.

### Results
Our models demonstrated robust performance in distinguishing between offensive and non-offensive comments, with nuanced differences in effectiveness across different datasets.
As you can see in the Jupeter notebook, Mistral won in the most aspects.


![WhatsApp Image 2024-04-08 at 14 00 40](https://github.com/Asanta4/LLMs---Offensive-or-not-/assets/136238984/f8a6c4fd-b193-43da-9410-478dede60d6c)
