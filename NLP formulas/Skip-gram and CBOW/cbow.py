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

# Разбиваем текст на слова и создаем частотный словарь
words = " ".join(corpus).split()
vocab = Counter(words)
word2idx = {word: i for i, word in enumerate(vocab.keys())}
idx2word = {i: word for word, i in word2idx.items()}
vocab_size = len(vocab)

# Генерация обучающих пар (контекстные слова -> центральное слово)
def generate_cbow_pairs(words, window_size=2):
    pairs = []
    for i in range(window_size, len(words) - window_size):
        context = [words[j] for j in range(i - window_size, i + window_size + 1) if j != i]
        target = words[i]
        pairs.append((context, target))
    return pairs

pairs = generate_cbow_pairs(words)

# === 2. Создание модели CBOW ===
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context_words):
        emb = self.embeddings(context_words)
        emb_sum = torch.sum(emb, dim=0)  # Усреднение векторов контекста
        out = self.linear(emb_sum)
        return out

# === 3. Обучение модели ===
embedding_dim = 10
context_size = 4  # Количество слов в контексте (по 2 слева и справа)
model = CBOW(vocab_size, embedding_dim, context_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Преобразование данных в тензоры
def context_to_tensor(context):
    return torch.tensor([word2idx[word] for word in context], dtype=torch.long)

def target_to_tensor(target):
    return torch.tensor([word2idx[target]], dtype=torch.long)

# Тренировка модели
num_epochs = 5000
for epoch in range(num_epochs):
    total_loss = 0
    for context, target in pairs:
        context_tensor = context_to_tensor(context)
        target_tensor = target_to_tensor(target)

        optimizer.zero_grad()
        output = model(context_tensor)
        loss = criterion(output.unsqueeze(0), target_tensor)
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

plt.title("Word2Vec CBOW Embeddings")
plt.show()
