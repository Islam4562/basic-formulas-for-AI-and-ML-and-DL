import torch
import torch.nn as nn
import torch.optim as optim

# === 1. Подготовка данных ===
corpus = ["hello how are you", "I am fine thank you", "how is your day", "it is great and sunny"]

# Создаем словарь
words = set(" ".join(corpus).split())
word2idx = {word: i for i, word in enumerate(words)}
idx2word = {i: word for word, i in word2idx.items()}
vocab_size = len(words)

# Генерация пар (входная последовательность -> следующее слово)
def create_sequences(corpus, seq_length=3):
    data = []
    for sentence in corpus:
        words = sentence.split()
        for i in range(len(words) - seq_length):
            input_seq = words[i:i+seq_length]
            target = words[i+seq_length]
            data.append((input_seq, target))
    return data

seq_length = 3
train_data = create_sequences(corpus, seq_length)

# Преобразование данных в тензоры
def encode_sequence(seq):
    return torch.tensor([word2idx[word] for word in seq], dtype=torch.long)

# === 2. Определение модели LSTM ===
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Берем выход последнего шага
        return out

# === 3. Обучение модели LSTM ===
embedding_dim = 10
hidden_dim = 20
model = LSTM(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 500
for epoch in range(num_epochs):
    total_loss = 0
    for seq, target in train_data:
        seq_tensor = encode_sequence(seq).unsqueeze(0)
        target_tensor = torch.tensor([word2idx[target]], dtype=torch.long)

        optimizer.zero_grad()
        output = model(seq_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# === 4. Генерация текста с LSTM ===
def generate_text(model, start_seq, length=5):
    model.eval()
    words = start_seq.split()
    for _ in range(length):
        seq_tensor = encode_sequence(words[-seq_length:]).unsqueeze(0)
        output = model(seq_tensor)
        predicted_word = idx2word[torch.argmax(output).item()]
        words.append(predicted_word)
    return " ".join(words)

print("\nGenerated text with LSTM:")
print(generate_text(model, "how is your"))
