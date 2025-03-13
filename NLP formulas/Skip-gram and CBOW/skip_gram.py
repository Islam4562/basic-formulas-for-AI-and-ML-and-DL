import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import random

# === 1. Подготовка данных ===
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the dog barked at the stranger",
    "the fox is clever and quick"
]

# Разбиваем текст на слова и считаем частоту слов
words = " ".join(corpus).split()
vocab = Counter(words)
word2idx = {word: i for i, word in enumerate(vocab.keys())}
idx2word = {i: word for word, i in word2idx.items()}
vocab_size = len(vocab)

# Генерация пар (центральное слово, контекстное слово)
def generate_skipgram_pairs(words, window_size=2):
    pairs = []
    for i, word in enumerate(words):
        for j in range(-window_size, window_size + 1):
            if j != 0 and 0 <= i + j < len(words):
                pairs.append((word, words[i + j]))
    return pairs

pairs = generate_skipgram_pairs(words)

# === 2. Создание модели Skip-gram ===
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_layer = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, center_word):
        emb = self.embeddings(center_word)
        out = self.out_layer(emb)
        return out

# === 3. Обучение модели ===
embedding_dim = 10
model = SkipGram(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Преобразование данных в тензоры
def word_to_tensor(word):
    return torch.tensor([word2idx[word]], dtype=torch.long)

# Тренировка модели
num_epochs = 5000
for epoch in range(num_epochs):
    total_loss = 0
    for center, context in pairs:
        center_tensor = word_to_tensor(center)
        context_tensor = torch.tensor([word2idx[context]], dtype=torch.long)

        optimizer.zero_grad()
        output = model(center_tensor)
        loss = criterion(output, context_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# === 4. Визуализация эмбеддингов ===
embeddings = model.embeddings.weight.data.numpy()

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 6))
for i, word in enumerate(word2idx.keys()):
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
    plt.text(reduced_embeddings[i, 0] + 0.1, reduced_embeddings[i, 1] + 0.1, word)

plt.title("Word2Vec Skip-gram Embeddings")
plt.show()
