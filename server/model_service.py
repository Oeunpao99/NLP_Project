# import os
# import json
# import re
# import numpy as np
# from typing import List, Dict, Any

# import torch
# import torch.nn as nn
# # torchcrf is optional for serving model predictions. If missing, the server will keep running
# # and predictions will fall back to the heuristic regex-based detector.
# try:
#     # try common import name
#     from torchcrf import CRF
#     HAVE_TORCHCRF = True
# except Exception:
#     try:
#         # some distributions expose a capitalized package name
#         from TorchCRF import CRF
#         HAVE_TORCHCRF = True
#     except Exception as e:
#         CRF = None
#         HAVE_TORCHCRF = False
#         print('Warning: torchcrf (or TorchCRF) not importable. Install it with `pip install torchcrf` or `pip install -r server/requirements.txt` to enable model-based predictions. Error:', e)

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "model")
# os.makedirs(MODEL_DIR, exist_ok=True)

# # --- Char Autoencoder (small version, compatible with training notebook) ---
# class CharAutoencoder(nn.Module):
#     def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, num_layers=2, dropout=0.2):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.encoder_gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
#         self.decoder_gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
#         self.output_fc = nn.Linear(hidden_size, vocab_size)
#         self.vocab_size = vocab_size
#         self.embedding_dim = embedding_dim
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#     def forward(self, x):
#         # x: (batch, seq_len)
#         embeds = self.embedding(x)
#         outputs, h = self.encoder_gru(embeds)
#         return outputs, h


# class BiLSTM_CRF(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, tagset_size):
#         super().__init__()
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
#         self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
#         # CRF may be unavailable or have different constructor signature at runtime; try to initialize gracefully
#         self.crf = None
#         if HAVE_TORCHCRF and CRF is not None:
#             try:
#                 # some CRF implementations accept batch_first
#                 self.crf = CRF(tagset_size, batch_first=True)
#             except TypeError:
#                 try:
#                     # other implementations take only num_labels
#                     self.crf = CRF(tagset_size)
#                 except Exception as e:
#                     print('Failed to initialize CRF:', e)
#                     self.crf = None
#             except Exception as e:
#                 print('Failed to initialize CRF:', e)
#                 self.crf = None

#     def forward(self, embeds):
#         lstm_out, _ = self.lstm(embeds)
#         emissions = self.hidden2tag(lstm_out)
#         return emissions

#     def loss(self, embeds, tags, mask):
#         if self.crf is None:
#             raise RuntimeError('CRF not available; cannot compute loss on server.')
#         emissions = self.forward(embeds)
#         # Support different CRF implementations: some expose forward(h, labels, mask) returning
#         # log-likelihood per batch, others implement __call__(emissions, tags, mask=..., reduction=..)
#         if hasattr(self.crf, 'forward'):
#             ll = self.crf.forward(emissions, tags, mask)
#             if isinstance(ll, torch.Tensor):
#                 return -ll.mean()
#             else:
#                 # unknown return, fallback
#                 return -torch.tensor(ll).mean()
#         elif callable(self.crf):
#             # older-style API
#             return -self.crf(emissions, tags, mask=mask, reduction='mean')
#         else:
#             raise RuntimeError('Unsupported CRF interface')

#     def predict(self, embeds, mask):
#         if self.crf is None:
#             raise RuntimeError('CRF not available; prediction cannot be performed by BiLSTM_CRF')
#         emissions = self.forward(embeds)
#         # support both decode and viterbi_decode
#         if hasattr(self.crf, 'decode'):
#             return self.crf.decode(emissions, mask=mask)
#         elif hasattr(self.crf, 'viterbi_decode'):
#             return self.crf.viterbi_decode(emissions, mask=mask)
#         else:
#             raise RuntimeError('Unsupported CRF interface for decoding')


# # --- Utility functions ---
# def simple_tokenize(text: str) -> List[str]:
#     # basic whitespace + punctuation split (keeps Khmer characters)
#     tokens = re.findall(r"[\u1780-\u17FF]+|\d+|[^\s\w]+", text, flags=re.UNICODE)
#     if not tokens:
#         # fallback: split on whitespace
#         return [t for t in text.split() if t]
#     return tokens


# def decode_bio(tokens: List[str], labels: List[str], label_map: Dict[int, str]) -> List[Dict[str, Any]]:
#     """Convert BIO-style tags to entity spans"""
#     entities = []
#     current = None
#     for i, (tok, tag) in enumerate(zip(tokens, labels)):
#         if tag == 'O' or tag.upper() == 'O':
#             if current:
#                 entities.append(current)
#                 current = None
#             continue
#         # tag like B-PER, I-LOC, or just PER
#         if tag.startswith('B-'):
#             if current:
#                 entities.append(current)
#             current = {'text': tok, 'type': tag[2:].lower(), 'start_token': i, 'end_token': i+1}
#         elif tag.startswith('I-'):
#             if current:
#                 current['text'] += ' ' + tok
#                 current['end_token'] = i+1
#             else:
#                 # malformed I- without B-: start new
#                 current = {'text': tok, 'type': tag[2:].lower(), 'start_token': i, 'end_token': i+1}
#         else:
#             # single label 'PER' treat as B-PER
#             if current:
#                 entities.append(current)
#             current = {'text': tok, 'type': tag.lower(), 'start_token': i, 'end_token': i+1}
#     if current:
#         entities.append(current)
#     # convert token indices to character positions (approximate)
#     char_pos = []
#     pos = 0
#     token_positions = []
#     for t in tokens:
#         start = pos
#         end = pos + len(t)
#         token_positions.append((start, end))
#         pos = end + 1  # assume one space
#     for ent in entities:
#         s_tok = ent['start_token']
#         e_tok = ent['end_token'] - 1
#         start_char = token_positions[s_tok][0] if s_tok < len(token_positions) else 0
#         end_char = token_positions[e_tok][1] if e_tok < len(token_positions) else start_char + len(ent['text'])
#         char_pos.append({'text': ent['text'], 'type': ent['type'], 'start': start_char, 'end': end_char})
#     return char_pos


# # --- Heuristic fallback (re-uses JS heuristics) ---
# _PERSON_RE = re.compile(r"(លោក|អ្នកស្រី|សម្តេច|ព្រះ|អ្នកឧកញ៉ា)\s+[\u1780-\u17FF]+(?:\s+[\u1780-\u17FF]+){0,2}")
# _NAME_RE = re.compile(r"([\u1780-\u17FF]+)\s+([\u1780-\u17FF]+)")
# _LOC_RE = re.compile(r"(ភ្នំពេញ|សៀមរាប|ព្រះសីហនុ|កំពង់ចាម|បាត់ដំបង)|(?:\b(?:ខេត្ត|រាជធានី|ក្រុង)\s+[\u1780-\u17FF]+)")
# _ORG_RE = re.compile(r"(ក្រុមហ៊ុន|ធនាគារ|អង្គការ|ក្រសួង|រដ្ឋសភា)\s+[\u1780-\u17FF]+(?:\s+[\u1780-\u17FF]+){0,3}")
# _DATE_RE = re.compile(r"ថ្ងៃទី\d+\s+ខែ[\u1780-\u17FF]+\s+ឆ្នាំ\d+|ខែ[\u1780-\u17FF]+\s+ឆ្នាំ\d+")
# _MONEY_RE = re.compile(r"\d+\s+(លាន|ពាន់|រយ)\s+ដុល្លារ|\d+\s+ដុល្លារ")


# def heuristic_entities(text: str) -> List[Dict[str, Any]]:
#     ents = []
#     for m in _PERSON_RE.finditer(text):
#         ents.append({'text': m.group(0), 'type': 'person', 'start': m.start(), 'end': m.end()})
#     for m in _LOC_RE.finditer(text):
#         ents.append({'text': m.group(0), 'type': 'location', 'start': m.start(), 'end': m.end()})
#     for m in _ORG_RE.finditer(text):
#         ents.append({'text': m.group(0), 'type': 'organization', 'start': m.start(), 'end': m.end()})
#     for m in _DATE_RE.finditer(text):
#         ents.append({'text': m.group(0), 'type': 'date', 'start': m.start(), 'end': m.end()})
#     for m in _MONEY_RE.finditer(text):
#         ents.append({'text': m.group(0), 'type': 'money', 'start': m.start(), 'end': m.end()})
#     # deduplicate on (text,start)
#     seen = set()
#     out = []
#     for e in sorted(ents, key=lambda x: x['start']):
#         key = (e['text'], e['start'])
#         if key in seen: continue
#         seen.add(key)
#         out.append(e)
#     return out


# class ModelService:
#     def __init__(self):
#         self.bilstm = None
#         self.autoencoder = None
#         self.idx2label = None
#         self.char2idx = None
#         self.ready = False
#         self.last_load_error = None
#         self.embedding_dim = 256
#         self.autoencoder_config = dict(embedding_dim=128, hidden_size=256, num_layers=2, dropout=0.2)
#         self._try_load_defaults()

#     def _try_load_defaults(self):
#         # Try to load files from MODEL_DIR
#         bilstm_path = os.path.join(MODEL_DIR, 'D:\I5_AMS_Accadamic/NLP TP/Project/NLP/server/model/bilstm_crf_best.pt')
#         auto_path = os.path.join(MODEL_DIR, 'char_autoencoder_epoch3.pt')
#         idx2label_path = os.path.join(MODEL_DIR, 'idx2label.json')
#         char2idx_path = os.path.join(MODEL_DIR, 'char2idx.json')

#         loaded = False
#         if os.path.exists(idx2label_path):
#             with open(idx2label_path, 'r', encoding='utf-8') as f:
#                 self.idx2label = json.load(f)
#         if os.path.exists(char2idx_path):
#             with open(char2idx_path, 'r', encoding='utf-8') as f:
#                 self.char2idx = json.load(f)

#         try:
#             if os.path.exists(bilstm_path):
#                 if not HAVE_TORCHCRF:
#                     print('torchcrf not available; skipping BiLSTM model load. Install torchcrf to enable model-based predictions.')
#                 else:
#                     # we need tagset size; if checkpoint contains final linear weights, prefer that size
#                     state = torch.load(bilstm_path, map_location='cpu')
#                     inferred_tagset = None
#                     if isinstance(state, dict):
#                         # try to get label count from hidden2tag weight in checkpoint
#                         for k in ('hidden2tag.weight', 'hidden2tag.weight_0', 'hidden2tag.weight0'):
#                             if k in state:
#                                 inferred_tagset = state[k].shape[0]
#                                 break
#                     tagset_size = None
#                     if inferred_tagset is not None:
#                         tagset_size = int(inferred_tagset)
#                     else:
#                         tagset_size = len(self.idx2label) if self.idx2label else 11
#                     self.bilstm = BiLSTM_CRF(embedding_dim=self.embedding_dim, hidden_dim=128, tagset_size=tagset_size)
#                     # state might be a state_dict or a full model object; handle both
#                     try:
#                         if isinstance(state, dict):
#                             try:
#                                 # try strict load first
#                                 self.bilstm.load_state_dict(state)
#                                 loaded = True
#                             except Exception as e_strict:
#                                 self.last_load_error = str(e_strict)
#                                 print('Strict load failed, trying non-strict:', e_strict)
#                                 try:
#                                     self.bilstm.load_state_dict(state, strict=False)
#                                     loaded = True
#                                 except Exception as e_nonstrict:
#                                     self.last_load_error = str(e_nonstrict)
#                                     print('Non-strict load also failed:', e_nonstrict)
#                         else:
#                             # maybe the whole model object was saved
#                             if isinstance(state, nn.Module):
#                                 self.bilstm = state
#                                 loaded = True
#                             else:
#                                 self.last_load_error = 'Loaded object is not a state_dict or nn.Module, skipping.'
#                                 print('Loaded object is not a state_dict or nn.Module, skipping.')
#                         if loaded:
#                             self.bilstm.eval()
#                     except Exception as e:
#                         self.last_load_error = str(e)
#                         print('Error while loading BiLSTM model:', e)
#         except Exception as e:
#             print('Failed to load BiLSTM model:', e)

#         try:
#             if os.path.exists(auto_path) and self.char2idx:
#                 cfg = self.autoencoder_config
#                 self.autoencoder = CharAutoencoder(len(self.char2idx), **cfg)
#                 self.autoencoder.load_state_dict(torch.load(auto_path, map_location='cpu'))
#                 self.autoencoder.eval()
#                 loaded = True
#         except Exception as e:
#             print('Failed to load autoencoder:', e)

#         # Consider model ready if bilstm is loaded and idx2label mapping is present
#         self.ready = (self.bilstm is not None) and (self.idx2label is not None)

#     def load_uploaded_files(self, files: Dict[str, bytes]):
#         # files: keys can be 'bilstm', 'autoencoder', 'idx2label', 'char2idx'
#         saved = []
#         for name, content in files.items():
#             path = os.path.join(MODEL_DIR, {
#                 'bilstm': 'D:\I5_AMS_Accadamic/NLP TP/Project/NLP/server/model/bilstm_crf_best.pt',
#                 'autoencoder': 'char_autoencoder_epoch3.pt',
#                 'idx2label': 'idx2label.json',
#                 'char2idx': 'char2idx.json'
#             }[name])
#             with open(path, 'wb') as f:
#                 f.write(content)
#             saved.append(path)
#         # reload
#         self._try_load_defaults()
#         return saved

#     def predict(self, text: str) -> Dict[str, Any]:
#         if not text or not text.strip():
#             return {'entities': []}
#         # If model loaded try to use it
#         if self.bilstm is not None and self.idx2label is not None and HAVE_TORCHCRF:
#             tokens = simple_tokenize(text)
#             # get embeddings
#             embs = []
#             for tok in tokens:
#                 vec = self._get_word_embedding(tok)
#                 embs.append(vec)
#             if len(embs) == 0:
#                 return {'entities': []}
#             x = torch.tensor(np.stack(embs, axis=0), dtype=torch.float32).unsqueeze(0)  # (1, seq, emb_dim)
#             mask = torch.ones((1, x.size(1)), dtype=torch.bool)
#             try:
#                 pred_ids = self.bilstm.predict(x, mask)[0]
#                 # map to labels
#                 labels = [self.idx2label.get(str(i), 'O') if isinstance(self.idx2label, dict) else self.idx2label[i] for i in pred_ids]
#                 entities = decode_bio(tokens, labels, {})
#                 return {'entities': entities, 'labels': labels}
#             except Exception as e:
#                 print('Model prediction failed:', e)
#                 # fallback
#         # Fallback heuristic
#         ents = heuristic_entities(text)
#         return {'entities': ents}

#     def info(self) -> Dict:
#         bilstm_path = os.path.join(MODEL_DIR, 'D:\I5_AMS_Accadamic/NLP TP/Project/NLP/server/model/bilstm_crf_best.pt')
#         auto_path = os.path.join(MODEL_DIR, 'char_autoencoder_epoch3.pt')
#         idx2label_path = os.path.join(MODEL_DIR, 'idx2label.json')
#         char2idx_path = os.path.join(MODEL_DIR, 'char2idx.json')
#         return {
#             'ready': self.ready,
#             'have_torchcrf': HAVE_TORCHCRF,
#             'bilstm_exists': os.path.exists(bilstm_path),
#             'autoencoder_exists': os.path.exists(auto_path),
#             'idx2label_exists': os.path.exists(idx2label_path),
#             'char2idx_exists': os.path.exists(char2idx_path),
#             'loaded_bilstm': self.bilstm is not None,
#             'loaded_autoencoder': self.autoencoder is not None,
#             'idx2label_loaded': self.idx2label is not None,
#             'char2idx_loaded': self.char2idx is not None,
#             'last_load_error': self.last_load_error
#         }
#     def _get_word_embedding(self, word: str):
#         # use autoencoder if available
#         if self.autoencoder is not None and self.char2idx is not None:
#             char_indices = [self.char2idx.get(c, self.char2idx.get('<PAD>', 0)) for c in word]
#             char_tensor = torch.tensor(char_indices, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
#             with torch.no_grad():
#                 embedded = self.autoencoder.embedding(char_tensor)
#                 _, hidden = self.autoencoder.encoder_gru(embedded)
#             word_emb = hidden[-1].squeeze(0).numpy()
#             return word_emb.astype(np.float32)
#         # deterministic fallback embedding using seeded RNG by word
#         rng = np.random.RandomState(abs(hash(word)) % (2**32))
#         return rng.normal(size=(self.embedding_dim,)).astype(np.float32)


# # Initialize a global instance
# service = ModelService()
