# 🧠 Transformer-Based Question Answering with RAG (C4)

This repository demonstrates a **Transformer-based question answering system** enhanced with **Retrieval-Augmented Generation (RAG)** using chunked C4 data. 
The system retrieves relevant context for a given question and generates answers with a custom encoder-decoder model built from scratch.

Inspired by the paper: *"Attention is All You Need"* (Vaswani et al., 2017) and guided with insights from ChatGPT.

# ✨ Key Educational Features

- Step-by-step integration of a **Transformer** from scratch
- Hands-on implementation of **multi-head attention** and **teacher forcing**
- Demonstrates **RAG pipeline**: chunking → retrieval → context-conditioned generation
- Uses **SentencePiece tokenizer** and shows tokenization for NLP tasks
- Fully reproducible training and inference pipelines
- Practical example for learning **TF-IDF retrieval** and **cosine similarity** for question answering

# 📁 Project Structure
```bash
transformer_model/
│
├── QuestionAnswering.py       # RAG-enabled QA inference script
├── main.py                    # Training & inference pipeline
├── Transformer.py             # Full Transformer model
├── Encoder.py                 # Encoder stack
├── Decoder.py                 # Decoder stack
├── DecoderLayer.py            # Masked attention decoder layer
├── helper.py                  # Masks, preprocessing, utilities
├── rag_utils.py               # TF-IDF retriever, chunking functions
├── corpus/                    # Optional datasets
├── data/                      # C4 JSONL & SQuAD datasets
├── models/                    # SentencePiece tokenizer, saved models
├── pretrained_models/         # Pretrained QA models
├── requirements.txt           # Dependencies
└── README.md                  # This file
```
# 🔗 RAG Component Overview

1. **Chunking C4 Data**:
   - Each large document is split into overlapping chunks (~300 words, 50-word overlap)
   - Ensures better retrieval granularity

2. **TF-IDF Retrieval**:
   - Each chunk is vectorized with **TfidfVectorizer**
   - Cosine similarity is computed between the question and all chunks
   - Top-k chunks are selected as relevant context

3. **Context-Conditioned Question Answering**:
   - The input to the transformer is formatted as:
     ```
     question: <your question> context: <retrieved chunks>
     ```
   - Transformer generates the answer based on retrieved context

4. **Greedy Decoding**:
   - Step-by-step token prediction until `<EOS>` token
   - Teacher forcing is used during training to improve learning

# 🏗 Transformer Architecture (Educational Focus)

## Encoder
- Token embedding + positional encoding
- Multi-head self-attention
- Feed-forward layers
- Stacked encoder blocks

## Decoder
- Masked self-attention for autoregressive prediction
- Encoder-decoder attention to integrate context
- Feed-forward layers
- Output softmax over the vocabulary

# ⚙ Training Configuration (Hands-On Learning)

- **Encoder max length**: 150 tokens  
- **Decoder max length**: 50 tokens  
- **Embedding dimension**: 128  
- **Number of layers**: 2  
- **Attention heads**: 2  
- **Batch size**: 64  
- **Epochs**: 20  
- **Optimizer**: Adam with warmup schedule  
- **Loss function**: Masked Sparse Categorical Crossentropy  

This configuration demonstrates practical Transformer tuning for educational purposes.

# 💾 Saving & Loading Models

- Save weights:
  ```python
  transformer.save_weights("transformer_weights.h5")
  ```
**Transformer Decoding**: The model generates answers conditioned on both the question and retrieved context.

# 🏗 Model Architecture

## Encoder
- Token embedding + positional encoding
- Stacked encoder layers
- Multi-head self-attention
- Feed-forward networks

## Decoder
- Masked self-attention (look-ahead mask)
- Encoder–decoder attention
- Feed-forward network
- Final softmax over vocabulary

# ⚙ Training Configuration

- Encoder max length: 150 tokens
- Decoder max length: 50 tokens
- Embedding dimension: 128
- Number of layers: 2
- Attention heads: 2
- Batch size: 64
- Epochs: 20
- Optimizer: Adam with custom warmup schedule
- Loss: Masked Sparse Categorical Crossentropy

# 💾 Saving & Loading Models

- Save model weights:
```python
transformer.save_weights("transformer_weights.h5")
```
- Load Model weights:
```python
transformer.load_weights("transformer_weights.h5")
```
# 🧩 Educational Notes

## Why RAG?
- Helps the model answer questions using large knowledge sources without storing all facts in parameters.

## Why chunking?
- Enables retrieval of context that fits into model input limits.

## TF-IDF vs embeddings
- TF-IDF provides a lightweight, interpretable retrieval mechanism before experimenting with dense embeddings.

## Hands-on insight
- Combines classical IR with modern sequence generation — ideal for educational exploration.

# 📌 Credits

- Paper: "Attention is All You Need", Vaswani et al., 2017
- Guidance & assistance from ChatGPT
- C4 dataset used for RAG retrieval experiments

# 🚀 Conclusion

- This repository is designed for learners to:
 1. Understand Transformer mechanics step by step
 2. Learn retrieval-based question answering (RAG)
 3. Experiment with chunked corpora and TF-IDF
 4. Observe context-conditioned text generation in action

