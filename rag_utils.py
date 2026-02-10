import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        if len(chunk) > 50:
            chunks.append(" ".join(chunk))
    return chunks


# Retriever
class TfidfRetriever:
    def __init__(self, documents, doc_vectors, vectorizer):
        self.documents = documents
        self.doc_vectors = doc_vectors
        self.vectorizer = vectorizer

    def retrieve(self, question, k=3):
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.doc_vectors)[0]
        top_k_idx = scores.argsort()[-k:][::-1]
        return [self.documents[i] for i in top_k_idx]
