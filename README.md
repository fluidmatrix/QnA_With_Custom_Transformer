# 🧠 Transformer-Based Text Summarization (TensorFlow)

An end-to-end implementation of an **abstractive text summarization model**
using a custom **Transformer (Encoder–Decoder)** architecture built with
TensorFlow and Keras.

--------------------------------------------------------------------------

## ✨ Features

- Custom Transformer implementation (no high-level shortcuts)
- Encoder–Decoder architecture with Multi-Head Attention
- Look-ahead and padding masks
- Teacher forcing during training
- Greedy decoding during inference
- SOS / EOS token-based sequence generation
- Fully reproducible training pipeline

--------------------------------------------------------------------------

## 📁 Project Structure
```bash
transformer_model/
│
├── main.py                # Training & inference pipeline
├── Transformer.py         # Full Transformer model
├── Encoder.py             # Encoder stack
├── Decoder.py             # Decoder stack
├── DecoderLayer.py        # Masked attention decoder layer
├── helper.py              # Masks, preprocessing, utilities
├── corpus/                # Training & test datasets
│   ├── train.json
│   └── test.json
├── requirements.txt       # Dependencies (pip freeze)
└── README.md
```
--------------------------------------------------------------------------

## 🏗 Model Architecture

### Encoder
- Token embedding + positional encoding
- Stacked encoder layers
- Multi-head self-attention
- Feed-forward networks

### Decoder
- Masked self-attention (look-ahead mask)
- Encoder–decoder attention (padding mask)
- Feed-forward network
- Final softmax over vocabulary

--------------------------------------------------------------------------

## ⚙ Training Configuration

### Training Loss
![PLT PLOT](<113638.png>)

### Sequence Lengths
- Encoder max length: 150
- Decoder max length: 50

### Hyperparameters
- Embedding dimension: 128
- Number of layers: 2
- Attention heads: 2
- Batch size: 64
- Epochs: 20

### Optimization
- Optimizer: Adam
- Learning rate: Custom warmup schedule
- Loss: Masked Sparse Categorical Crossentropy

--------------------------------------------------------------------------

## 📉 Loss Function

- Padding tokens are ignored
- Loss is computed only on valid tokens
- Normalized by number of non-padding tokens

--------------------------------------------------------------------------

## 🔍 Inference & Summarization

### How Inference Works
1. Encode the input document
2. Initialize decoder with [SOS]
3. Predict tokens step-by-step
4. Stop at [EOS] or max decoder length

### Example

Input:
[SOS] amanda: i baked cookies... [EOS]

Human Summary:
[SOS] amanda baked cookies and will bring jerry some tomorrow. [EOS]

Model Output:
Generated using greedy decoding

--------------------------------------------------------------------------

## ▶ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 2️⃣ Prepare Dataset

Place your dataset inside:

./corpus/

### 3️⃣ Train the Model
```bash
python main.py
```
During training, the script will:
- Print batch-wise progress
- Report loss after each epoch
- Show example summarizations from the test set

--------------------------------------------------------------------------

## 💾 Saving & Loading the Model

### Save model weights:
```bash
transformer.save_weights("transformer_weights.h5")
```
### Load model weights:
```bash
transformer.load_weights("transformer_weights.h5")
```
### Once loaded, you can directly generate summaries:
```bash
summarize(transformer, input_text)
```
--------------------------------------------------------------------------

## 📌 Notes

- The model does NOT need retraining if weights are saved and reloaded
- Inference runs with training=False
- Designed for clarity and educational purposes
