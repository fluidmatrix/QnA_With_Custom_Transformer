import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import textwrap
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
from termcolor import colored
from rag_utils import TfidfRetriever, chunk_text
from helper import get_sentinels, parse_squad, answer_question, pretty_decode
import transformer_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Setup
# ----------------------------
wrapper = textwrap.TextWrapper(width=70)
np.random.seed(42)

# ----------------------------
# Load C4 text data
# ----------------------------
with open('data/c4-en-10k.jsonl', 'r') as file:
    example_jsons = [json.loads(line.strip()) for line in file]

natural_language_texts = [example_json['text'] for example_json in example_jsons]

# ----------------------------
# Load tokenizer
# ----------------------------
with open("./models/sentencepiece.model", "rb") as f:
    pre_trained_tokenizer = f.read()
tokenizer = tf_text.SentencepieceTokenizer(pre_trained_tokenizer, out_type=tf.int32)
eos = tokenizer.string_to_id("</s>").numpy()
sentinels = get_sentinels(tokenizer, display=False)

# ----------------------------
# Define transformer model
# ----------------------------
num_layers = 2
embedding_dim = 128
fully_connected_dim = 128
num_heads = 2
positional_encoding_length = 256
encoder_vocab_size = int(tokenizer.vocab_size())
decoder_vocab_size = encoder_vocab_size

transformer = transformer_utils.Transformer(
    num_layers,
    embedding_dim,
    num_heads,
    fully_connected_dim,
    encoder_vocab_size,
    decoder_vocab_size,
    positional_encoding_length,
    positional_encoding_length,
)

# ----------------------------
# Load pretrained weights
# ----------------------------
transformer.load_weights('./pretrained_models/model_qa3').expect_partial()

# ----------------------------
# Load SQuAD data for evaluation
# ----------------------------
with open('data/train-v2.0.json', 'r') as f:
    squad_json = json.load(f)
inputs, targets = parse_squad(squad_json['data'])

inputs_test = inputs[40000:45000]
targets_test = targets[40000:45000]

# ----------------------------
# Build RAG document store
# ----------------------------
# Build chunks once
c4_documents = []
for text in natural_language_texts:
    c4_documents.extend(chunk_text(text))

print(f"[RAG] Total C4 chunks: {len(c4_documents)}")

# Build TF-IDF matrix once
vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1,2))
doc_vectors = vectorizer.fit_transform(c4_documents)


# ----------------------------
# RAG-enabled QA
# ----------------------------
retriever = TfidfRetriever(c4_documents, doc_vectors, vectorizer)

def answer_question_rag(question, transformer, tokenizer, retriever, k=3):
    top_chunks = retriever.retrieve(question, k=k)
    context = " ".join([chunk.strip() for chunk in top_chunks])
    model_input = f"question: {question} context: {context}"
    return answer_question(model_input, transformer, tokenizer)



# ----------------------------
# Run example inference
# ----------------------------
idx = 101
question = inputs_test[idx]
expected = targets_test[idx]

result = answer_question_rag(question, transformer, tokenizer, retriever, k=1)

predicted_answer = pretty_decode(result, sentinels, tokenizer).numpy()[0]

print(colored("Predicted Answer: " + str(predicted_answer), 'blue'))
print(colored("Question: " + str(question), 'grey'))
print(colored("Expected Answer: " + str(expected), 'green'))
