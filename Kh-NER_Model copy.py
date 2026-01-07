#!/usr/bin/env python
# coding: utf-8

# In[32]:


import os
import glob
import re
import math
import random
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchcrf import CRF
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# In[9]:


device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

# In[10]:


#load data
def load_conll(filename):
    sentences, labels = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words, tags = [], []
            else:
                word, tag = line.split('\t')
                words.append(word)
                tags.append(tag)
    return sentences, labels

train_sents, train_labels = load_conll('/Users/macbookair/Documents/Data Science 5th Year/Natural Language Processing-NLP/ner_train.conll')
dev_sents, dev_labels = load_conll('/Users/macbookair/Documents/Data Science 5th Year/Natural Language Processing-NLP/ner_dev.conll')
test_sents, test_labels = load_conll('/Users/macbookair/Documents/Data Science 5th Year/Natural Language Processing-NLP/ner_test.conll')

print("Train:", len(train_sents), "Dev:", len(dev_sents), "Test:", len(test_sents))

# In[11]:


all_labels = [l for sent in (train_labels + dev_labels + test_labels) for l in sent]
label_set = sorted(set(all_labels))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}
print("labels:", label_set)

# In[12]:


# Count label frequencies in all data
from collections import Counter

label_counts = Counter(all_labels)

print("Label frequencies:")
for label, count in label_counts.most_common():
    print(f"  {label:6}: {count:,}")

# In[13]:


from collections import Counter
import torch

# Compute label frequencies
all_labels = [l for sent in (train_labels + dev_labels + test_labels) for l in sent]
label_set = sorted(set(all_labels))
label2idx = {l: i for i, l in enumerate(label_set)}

label_counts = Counter(all_labels)
total = sum(label_counts.values())
class_weights = []

for label in label_set:
    # Inverse frequency weighting
    freq = label_counts[label] / total
    class_weights.append(1.0 / freq)

# Normalize weights
weights = torch.tensor(class_weights, dtype=torch.float)
weights = weights / weights.sum()

# Use in loss function
loss_fn = nn.CrossEntropyLoss(weight=weights)

# In[14]:


# Load the trained character-level autoencoder model
class CharAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.decoder_gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.output_fc = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, input_seq, target_seq=None, teacher_forcing_ratio=0.5):
        embedded = self.embedding(input_seq)
        _, hidden = self.encoder_gru(embedded)
        batch_size, seq_len = input_seq.size()
        decoder_input = input_seq[:, 0].unsqueeze(1)
        outputs = []
        decoder_hidden = hidden
        for t in range(seq_len):
            decoder_embedded = self.embedding(decoder_input)
            decoder_output, decoder_hidden = self.decoder_gru(decoder_embedded, decoder_hidden)
            logits = self.output_fc(decoder_output.squeeze(1))
            outputs.append(logits.unsqueeze(1))
            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                next_input = target_seq[:, t].unsqueeze(1)
            else:
                next_input = logits.argmax(dim=1, keepdim=True)
            decoder_input = next_input
        outputs = torch.cat(outputs, dim=1)
        return outputs

# In[15]:


# Paste your char2idx mapping here
char2idx = {
    '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, 'ក': 3, 'ខ': 4, 'គ': 5, 'ឃ': 6, 'ង': 7, 'ច': 8, 'ឆ': 9, 'ជ': 10, 'ឈ': 11, 'ញ': 12, 'ដ': 13, 'ឋ': 14, 'ឌ': 15, 'ឍ': 16, 'ណ': 17, 'ត': 18, 'ថ': 19, 'ទ': 20, 'ធ': 21, 'ន': 22, 'ប': 23, 'ផ': 24, 'ព': 25, 'ភ': 26, 'ម': 27, 'យ': 28, 'រ': 29, 'ល': 30, 'វ': 31, 'ឝ': 32, 'ឞ': 33, 'ស': 34, 'ហ': 35, 'ឡ': 36, 'អ': 37, 'ឣ': 38, 'ឤ': 39, 'ឥ': 40, 'ឦ': 41, 'ឧ': 42, 'ឩ': 43, 'ឪ': 44, 'ឫ': 45, 'ឬ': 46, 'ឭ': 47, 'ឮ': 48, 'ឯ': 49, 'ឰ': 50, 'ឱ': 51, 'ឲ': 52, 'ឳ': 53, '឵': 54, 'ា': 55, 'ិ': 56, 'ី': 57, 'ឹ': 58, 'ឺ': 59, 'ុ': 60, 'ូ': 61, 'ួ': 62, 'ើ': 63, 'ឿ': 64, 'ៀ': 65, 'េ': 66, 'ែ': 67, 'ៃ': 68, 'ោ': 69, 'ៅ': 70, 'ំ': 71, 'ះ': 72, 'ៈ': 73, '៉': 74, '៊': 75, '់': 76, '៌': 77, '៍': 78, '៎': 79, '៏': 80, '័': 81, '៑': 82, '្': 83, '៓': 84, '៖': 85, 'ៗ': 86, '៛': 87, '៝': 88, '០': 89, '១': 90, '២': 91, '៣': 92, '៤': 93, '៥': 94, '៦': 95, '៧': 96, '៨': 97, '៩': 98
}

embedding_dim = 128      
hidden_size = 256        
num_layers = 2           
dropout = 0.2            
vocab_size = len(char2idx)
autoencoder = CharAutoencoder(vocab_size, embedding_dim, hidden_size, num_layers, dropout)
autoencoder.load_state_dict(torch.load("/Users/macbookair/Documents/Data Science 5th Year/Natural Language Processing-NLP/char_autoencoder_epoch3.pt", map_location=device))
autoencoder.eval()
autoencoder.to(device)

# In[16]:


# Convert sentences to embeddings and labels to indices using autoencoder
def get_word_embedding(word, char2idx, autoencoder, device):
    char_indices = [char2idx.get(c, char2idx['<PAD>']) for c in word]
    char_tensor = torch.tensor(char_indices, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        embedded = autoencoder.embedding(char_tensor)
        _, hidden = autoencoder.encoder_gru(embedded)
    word_emb = hidden[-1].squeeze(0)  # [hidden_size]
    return word_emb.cpu().numpy()

def prepare_data(sentences, labels, char2idx, autoencoder, device, label2idx):
    X, y = [], []
    for sent, label_seq in zip(sentences, labels):
        word_embs = [get_word_embedding(word, char2idx, autoencoder, device) for word in sent]
        label_ids = [label2idx[l] for l in label_seq]
        X.append(word_embs)
        y.append(label_ids)
    return X, y

# Prepare train/dev/test sets
X_train, y_train = prepare_data(train_sents, train_labels, char2idx, autoencoder, device, label2idx)
X_dev, y_dev = prepare_data(dev_sents, dev_labels, char2idx, autoencoder, device, label2idx)
X_test, y_test = prepare_data(test_sents, test_labels, char2idx, autoencoder, device, label2idx)


# In[20]:


lengths = [len(sent) for sent in train_sents]
print("Min length:", min(lengths))
print("Max length:", max(lengths))
print("Average length:", sum(lengths)/len(lengths))

# In[21]:


# Count total words in all splits
total_words = sum(len(sent) for sent in train_sents + dev_sents + test_sents)
print("Total words:", total_words)

# Preview a few sentences and their labels
for i in range(3):  # Show first 3 sentences
    print("Sentence:", train_sents[i])
    print("Labels:  ", train_labels[i])
    print("---")

# In[22]:


print("First sentence:", train_sents[0])
print("First sentence embeddings (shape):", [emb.shape for emb in X_train[0]])
for i, (word, emb) in enumerate(zip(train_sents[0], X_train[0])):
    print(f"Word: {word}")
    print(f"Embedding vector (first 10 values): {emb[:10]}")
    print(f"Vector shape: {emb.shape}")
    print("---")

# In[24]:


#DataLoader and batching for variable-length sentences
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NERDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # List of [seq_len, emb_dim] arrays
        self.y = y  # List of [seq_len] label lists
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return np.array(self.X[idx]), np.array(self.y[idx])

def pad_collate(batch):
    # batch: list of (X, y) tuples
    Xs, ys = zip(*batch)
    lengths = [len(x) for x in Xs]
    max_len = max(lengths)
    emb_dim = Xs[0].shape[1]
    # Pad X
    X_padded = np.zeros((len(Xs), max_len, emb_dim), dtype=np.float32)
    for i, x in enumerate(Xs):
        X_padded[i, :len(x), :] = x
    # Pad y
    y_padded = np.full((len(ys), max_len), fill_value=-1, dtype=np.int64)  # -1 for ignore index
    for i, y in enumerate(ys):
        y_padded[i, :len(y)] = y
    # Mask: 1 for real tokens, 0 for pad
    mask = (y_padded != -1)
    return (
        torch.tensor(X_padded),
        torch.tensor(y_padded),
        torch.tensor(mask, dtype=torch.bool)
    )

# In[27]:


#Training
train_dataset = NERDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)

#validation/test:
dev_dataset = NERDataset(X_dev, y_dev)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)
test_dataset = NERDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)

# Inspect a batch
for X_batch, y_batch, mask in train_loader:
    print("X_batch shape:", X_batch.shape)  # (batch, max_len, emb_dim)
    print("y_batch shape:", y_batch.shape)  # (batch, max_len)
    print("mask shape:", mask.shape)        # (batch, max_len)
    print("First mask row:", mask[0])
    break

# # NER Model 

# In[48]:


class BiLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, embeds):
        # embeds: (batch, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def loss(self, embeds, tags, mask):
        emissions = self.forward(embeds)
        return -self.crf(emissions, tags, mask=mask, reduction='mean')

    def predict(self, embeds, mask):
        emissions = self.forward(embeds)
        return self.crf.decode(emissions, mask=mask)

# In[49]:


#instantiation:
tagset_size = len(label2idx)
hidden_dim = 128
embedding_dim = 256 

model = BiLSTM_CRF_Direct(
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    tagset_size=tagset_size
).to(device)

print(model)

# In[31]:


num_epochs = 10
learning_rate = 0.001

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
best_val_loss = float('inf')

# In[33]:


# Training 
train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch, mask in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        loss = model.loss(X_batch, y_batch, mask)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch, mask in tqdm(dev_loader, desc=f"Epoch {epoch} - Validation"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask = mask.to(device)
            loss = model.loss(X_batch, y_batch, mask)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(dev_loader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "bilstm_crf_best.pt")
        print("Best model saved!")

print("Training complete.")

# In[34]:


# Visualize training and validation loss curves
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
epochs = range(1, len(train_losses) + 1)
sns.lineplot(x=epochs, y=train_losses, label='Train Loss')
sns.lineplot(x=epochs, y=val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True)
plt.show()

# In[35]:


# Load best model
model.load_state_dict(torch.load("bilstm_crf_best.pt", map_location=device))
model.eval()

# In[36]:


# Evaluate on test set and print classification report
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

y_true = []
y_pred = []

with torch.no_grad():
    for X_batch, y_batch, mask in tqdm(test_loader, desc="Testing"):
        X_batch = X_batch.to(device)
        y_batch = y_batch.cpu().numpy()
        mask = mask.cpu().numpy()
        preds = model.predict(X_batch, torch.tensor(mask, dtype=torch.bool, device=device))
        for i in range(len(preds)):
            true_labels = y_batch[i][mask[i]]
            pred_labels = preds[i]
            y_true.extend(true_labels)
            y_pred.extend(pred_labels)

# Remove ignored indices if any
valid_idx = [i for i, t in enumerate(y_true) if t != -1]
y_true = np.array(y_true)[valid_idx]
y_pred = np.array(y_pred)[valid_idx]

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# Classification report
print(classification_report(y_true, y_pred, target_names=[idx2label[i] for i in range(len(label2idx))]))

# In[38]:


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[idx2label[i] for i in range(len(label2idx))], yticklabels=[idx2label[i] for i in range(len(label2idx))])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for NER Labels')
plt.show()

# In[44]:


def predict_sentence_ner(sentence, model, char2idx, autoencoder, idx2label, device="cpu"):
    model.eval()
    tokens = sentence.strip().split()
    # Get char-level embeddings for each word
    word_embs = [get_word_embedding(word, char2idx, autoencoder, device) for word in tokens]
    X = torch.tensor([word_embs], dtype=torch.float32).to(device)  # [1, seq_len, emb_dim]
    mask = torch.ones(1, len(tokens), dtype=torch.bool).to(device)
    with torch.no_grad():
        pred_ids = model.predict(X, mask)[0]
    pred_labels = [idx2label[i] for i in pred_ids]
    for w, tag in zip(tokens, pred_labels):
        print(f"{w:<20s} {tag}")

# # Example 

# In[41]:


sentence = "ឧត្តមសេនីយ៍ឯក ជួន ណារិន្ទ ដឹកនាំ បើក កិច្ចប្រជុំ បូកសរុបលទ្ធផលប្រតិបត្តិការរក្សាសន្តិសុខ សណ្តាប់ធ្នាប់ សាធារណៈ សុវត្ថិភាព សង្គម ឆ្នាំ ២០២៥ និង លើក ទិសដៅ របស់ ស្នងការដ្ឋាននគរបាល រាជធានី ភ្នំពេញ"
predict_sentence_ner(sentence, model, char2idx, autoencoder, idx2label, device)

# In[45]:


sentence = "ថ្នាក់ដឹកនាំ​ ខេត្តត្បូងឃ្មុំ ដាក់ កម្រងផ្កា គោរពវិញ្ញាណក្ខន្ធសព ឧត្តមសេនីយ៍ត្រី សុខ ផល ដែល ពលី ជីវិត នៅ សមរភូមិមុខ ក្នុង បេសកកម្ម ការពារ ទឹកដី"
predict_sentence_ner(sentence, model, char2idx, autoencoder, idx2label, device)

# In[46]:



sentence = "មណីយដ្ឋានធម្មជាតិ ឆ្នេរខ្សាច់កោះប៉ែន និង ស្ពានប្រដឺសឬស្សី១.០០០ម៉ែត្រ សូមស្វាគមន៍ភ្ញៀវទេសចរណ៍"
predict_sentence_ner(sentence, model, char2idx, autoencoder, idx2label, device)

# In[47]:


sentence = "អំណោយរបស់ សម្តេចបវរធិបតី ហ៊ុន ម៉ាណែត និង លោកជំទាវបណ្ឌិត ត្រូវបាន បណ្ឌិត គុណ ញឹម បន្តនាំយកទៅប្រគល់ជូនដល់វីរកងទ័ពជួរមុខ ដែល កំពុង ឈរជើង ការពារ បូរណភាពទឹកដី តាម បណ្តោយ ព្រំដែន កម្ពុជា-ថៃ"
predict_sentence_ner(sentence, model, char2idx, autoencoder, idx2label, device)
