import json
import torch
import torch.nn.functional as F
from ml.ner_predictor import KhmerNERPredictor

# Sample sentence (from your notebook)
sentence = "អំណោយរបស់ សម្តេចបវរធិបតី ហ៊ុន ម៉ាណែត និង លោកជំទាវបណ្ឌិត ត្រូវបាន បណ្ឌិត គុណ ញឹម បន្តនាំយកទៅប្រគល់ជូនដល់វីរកងទ័ពជួរមុខ ដែល កំពុង ឈរជើង ការពារ បូរណភាពទឹកដី តាម បណ្តោយ ព្រំដែន កម្ពុជា-ថៃ"

p = KhmerNERPredictor('ml/model')
print('device:', p.device)
print('tagset_size:', p.tagset_size)
print('idx2label (sample):', json.dumps({str(k): v for k, v in p.idx2label.items()}, ensure_ascii=False))

# Tokenize (current runtime uses simple whitespace split)
tokens = sentence.strip().split()
print('\nTokens (whitespace split):')
for t in tokens:
    print(' -', t)

# Get embeddings
embs = [p.get_word_embedding(t) for t in tokens]
X = torch.tensor([embs], dtype=torch.float32).to(p.device)
mask = torch.ones(1, len(tokens), dtype=torch.bool).to(p.device)

if p.model is None:
    print('\nModel object is None (CRF not available). No emissions to inspect.')
else:
    with torch.no_grad():
        emissions = p.model.forward(X)  # shape: [1, seq_len, tagset]
        probs = F.softmax(emissions, dim=-1).cpu().numpy()[0]
        pred_ids = p.model.predict(X, mask)[0]
        pred_labels = [p.idx2label[i] for i in pred_ids]

    print('\nPer-token predictions and top probabilities:')
    for i, tok in enumerate(tokens):
        top_idx = probs[i].argsort()[-3:][::-1]
        top = [(int(j), p.idx2label[int(j)], float(probs[i, int(j)])) for j in top_idx]
        print(f"{i:02d}: {tok} -> pred_id={pred_ids[i]} label={pred_labels[i]} top={top}")

    # Show raw emissions for first token
    print('\nFirst token emissions (first 10 values):', emissions[0,0,:10].cpu().numpy())

print('\nDone')
