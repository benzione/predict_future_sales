import numpy as np

# Sample data (a small dataset for illustration purposes)
documents = [
    "I love coding in Python",
    "Natural Language Processing is fun",
    "Python is a versatile language",
    "NLP helps in understanding text",
]

# Helper function to preprocess text
def preprocess_text(text):
    return text.lower().split()

# Initialize hyperparameters
embedding_size = 10
learning_rate = 0.01
epochs = 100

# Create vocabulary
vocab = set()
for doc in documents:
    vocab.update(preprocess_text(doc))
vocab = list(vocab)
vocab_size = len(vocab)

# Initialize word and document embeddings randomly
word_embeddings = np.random.randn(vocab_size, embedding_size)
doc_embeddings = np.random.randn(len(documents), embedding_size)

# Training the Doc2Vec model
for epoch in range(epochs):
    for i, doc in enumerate(documents):
        doc_tokens = preprocess_text(doc)
        doc_index = i

        # Initialize the context target to the document itself
        context_targets = [doc_index]

        # Sample a random context from the document
        for j in range(len(doc_tokens)):
            target = vocab.index(doc_tokens[j])
            context = np.random.randint(len(doc_tokens))
            context_targets.append(target)

        # Update document and word embeddings
        doc_embedding = np.mean(word_embeddings[context_targets], axis=0)
        doc_embeddings[doc_index] = doc_embedding

        for target in context_targets:
            error = word_embeddings[target] - doc_embedding
            word_embeddings[target] -= learning_rate * error

# Example usage to retrieve document embeddings
for i, doc in enumerate(documents):
    print(f"Document {i + 1} Embedding:")
    print(doc_embeddings[i])
    print()