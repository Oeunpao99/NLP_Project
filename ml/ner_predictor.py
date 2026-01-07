"""
NER Model Wrapper for Khmer Language
"""

import torch
import torch.nn as nn
# Try to import available CRF package (common names: torchcrf or TorchCRF)
try:
    from torchcrf import CRF
except Exception:
    try:
        from TorchCRF import CRF  # some installs use capitalized package name
    except Exception:
        CRF = None

# Flag for convenience
HAVE_TORCHCRF = CRF is not None
import numpy as np
from typing import List, Tuple, Dict
import json
import logging

logger = logging.getLogger(__name__)

# Character mappings from your model
CHAR2IDX = {
    '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, 'ក': 3, 'ខ': 4, 'គ': 5, 'ឃ': 6, 'ង': 7, 'ច': 8, 'ឆ': 9, 'ជ': 10, 'ឈ': 11, 'ញ': 12, 'ដ': 13, 'ឋ': 14, 'ឌ': 15, 'ឍ': 16, 'ណ': 17, 'ត': 18, 'ថ': 19, 'ទ': 20, 'ធ': 21, 'ន': 22, 'ប': 23, 'ផ': 24, 'ព': 25, 'ភ': 26, 'ម': 27, 'យ': 28, 'រ': 29, 'ល': 30, 'វ': 31, 'ឝ': 32, 'ឞ': 33, 'ស': 34, 'ហ': 35, 'ឡ': 36, 'អ': 37, 'ឣ': 38, 'ឤ': 39, 'ឥ': 40, 'ឦ': 41, 'ឧ': 42, 'ឩ': 43, 'ឪ': 44, 'ឫ': 45, 'ឬ': 46, 'ឭ': 47, 'ឮ': 48, 'ឯ': 49, 'ឰ': 50, 'ឱ': 51, 'ឲ': 52, 'ឳ': 53, '឵': 54, 'ា': 55, 'ិ': 56, 'ី': 57, 'ឹ': 58, 'ឺ': 59, 'ុ': 60, 'ូ': 61, 'ួ': 62, 'ើ': 63, 'ឿ': 64, 'ៀ': 65, 'េ': 66, 'ែ': 67, 'ៃ': 68, 'ោ': 69, 'ៅ': 70, 'ំ': 71, 'ះ': 72, 'ៈ': 73, '៉': 74, '៊': 75, '់': 76, '៌': 77, '៍': 78, '៎': 79, '៏': 80, '័': 81, '៑': 82, '្': 83, '៓': 84, '៖': 85, 'ៗ': 86, '៛': 87, '៝': 88, '០': 89, '១': 90, '២': 91, '៣': 92, '៤': 93, '៥': 94, '៦': 95, '៧': 96, '៨': 97, '៩': 98
}

class CharAutoencoder(nn.Module):
    """Character-level autoencoder for word embeddings"""
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

class BiLSTM_CRF(nn.Module):
    """BiLSTM-CRF model for NER"""
    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        if CRF is None:
            raise ImportError("No CRF implementation found. Please install 'torchcrf' (pip install torchcrf) or an equivalent package.")
        # Try to construct CRF with common signatures
        try:
            self.crf = CRF(tagset_size, batch_first=True)
        except TypeError:
            # Fallback to constructor without batch_first
            self.crf = CRF(tagset_size)

    def forward(self, embeds):
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def loss(self, embeds, tags, mask):
        emissions = self.forward(embeds)
        # Try common modern API
        try:
            ll = self.crf(emissions, tags, mask=mask, reduction='mean')
            return -ll
        except TypeError:
            # Older API styles
            try:
                ll = self.crf.forward(emissions, tags, mask)
                return -ll
            except Exception as e:
                raise RuntimeError(f"CRF loss call failed: {e}")

    def predict(self, embeds, mask):
        emissions = self.forward(embeds)
        # Try different decode APIs
        try:
            return self.crf.decode(emissions, mask=mask)
        except Exception:
            try:
                return self.crf.viterbi_decode(emissions, mask)
            except Exception as e:
                raise RuntimeError(f"CRF decode failed: {e}")

class KhmerNERPredictor:
    """Main NER predictor class"""
    
    def __init__(self, model_dir: str = "ml/model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load label mappings
        with open(f"{model_dir}/label_mappings.json", "r", encoding="utf-8") as f:
            label_data = json.load(f)
        self.label2idx = label_data["label2idx"]
        self.idx2label = {int(k): v for k, v in label_data["idx2label"].items()}
        self.tagset_size = len(self.label2idx)
        
        # Initialize models
        self.char2idx = CHAR2IDX
        self.vocab_size = len(self.char2idx)
        self.embedding_dim = 128
        self.hidden_size = 256
        self.lstm_hidden_dim = 128
        
        # Load autoencoder
        self.autoencoder = CharAutoencoder(
            self.vocab_size, 
            self.embedding_dim, 
            self.hidden_size
        ).to(self.device)
        self.autoencoder.load_state_dict(
            torch.load(f"{model_dir}/char_autoencoder_epoch3.pt", map_location=self.device)
        )
        self.autoencoder.eval()
        
        # Load NER model — infer tagset size from checkpoint when possible
        ckpt_path = f"{model_dir}/bilstm_crf_best.pt"
        sd = None
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # Model might be saved directly as state_dict or wrapped
            if isinstance(ckpt, dict) and ('state_dict' in ckpt or 'model_state_dict' in ckpt):
                sd = ckpt.get('state_dict', ckpt.get('model_state_dict'))
            elif isinstance(ckpt, dict):
                sd = ckpt
            else:
                # Could be a full model object
                try:
                    sd = ckpt.state_dict()
                except Exception:
                    sd = None
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        inferred_tagset = None
        if sd is not None:
            # Look for hidden2tag weight or CRF transitions to infer tagset size
            if 'hidden2tag.weight' in sd:
                inferred_tagset = sd['hidden2tag.weight'].shape[0]
            elif 'hidden2tag.bias' in sd:
                inferred_tagset = sd['hidden2tag.bias'].shape[0]
            else:
                # Try CRF params
                for key in ('crf.transitions', 'crf.trans_matrix', 'crf.transitions.weight'):
                    if key in sd:
                        inferred_tagset = sd[key].shape[0]
                        break

        if inferred_tagset is not None and inferred_tagset != self.tagset_size:
            logger.warning(f"Label count mismatch: label mappings define {self.tagset_size} labels but checkpoint appears to use {inferred_tagset}. Using checkpoint tagset size to build the model.")
            # If label mapping size mismatches, replace idx2label with placeholders to avoid indexing errors
            if len(self.idx2label) != inferred_tagset:
                logger.warning("idx2label size does not match checkpoint; creating placeholder idx2label mapping. Please export your label mapping from the training environment and place it in model_dir.")
                self.idx2label = {i: f"LABEL_{i}" for i in range(inferred_tagset)}
            self.tagset_size = inferred_tagset

        # Initialize or skip model depending on CRF availability
        if not HAVE_TORCHCRF:
            logger.warning("torchcrf not available; skipping BiLSTM+CRF model load. Install the 'torchcrf' package to enable model-based predictions.")
            self.model = None
            logger.info("Models loaded (autoencoder present, BiLSTM+CRF skipped)")
        else:
            # Initialize model using reconciled tagset size
            self.model = BiLSTM_CRF(
                embedding_dim=self.hidden_size,
                hidden_dim=self.lstm_hidden_dim,
                tagset_size=self.tagset_size
            ).to(self.device)

            # Prepare state dict for loading: attempt to remap common CRF key names when necessary
            def _remap_crf_keys(ckpt_sd, model_sd):
                remapped = dict(ckpt_sd)
                # Common key name variants to try
                key_map_candidates = [
                    ("crf.start_transitions", "crf.start_trans"),
                    ("crf.end_transitions", "crf.end_trans"),
                    ("crf.transitions", "crf.trans_matrix"),
                    ("crf.transitions", "crf.transitions"),
                    ("crf.start_trans", "crf.start_transitions"),
                    ("crf.end_trans", "crf.end_transitions"),
                ]
                for src, dst in key_map_candidates:
                    if src in ckpt_sd and dst in model_sd and src not in remapped:
                        # copy if shapes match
                        if ckpt_sd[src].shape == model_sd[dst].shape:
                            remapped[dst] = ckpt_sd[src]
                return remapped

            loaded_ok = False
            load_errors = []
            try:
                # Try strict load first
                self.model.load_state_dict(sd)
                loaded_ok = True
            except Exception as e:
                load_errors.append(str(e))
                logger.warning("Strict load failed; attempting to remap CRF keys and load non-strictly.")
                try:
                    model_sd = self.model.state_dict()
                    remapped = _remap_crf_keys(sd if sd is not None else {}, model_sd)
                    res = self.model.load_state_dict(remapped, strict=False)
                    # res is a NamedTuple with missing_keys and unexpected_keys (PyTorch >=1.6)
                    if hasattr(res, 'missing_keys') or hasattr(res, 'unexpected_keys'):
                        load_errors.append(f"missing_keys={res.missing_keys} unexpected_keys={res.unexpected_keys}")
                    loaded_ok = True
                except Exception as e2:
                    load_errors.append(str(e2))
                    loaded_ok = False

            if not loaded_ok:
                err_msg = "; ".join(load_errors)
                raise RuntimeError(f"Failed to load model checkpoint cleanly: {err_msg}")

            self.model.to(self.device)
            self.model.eval()

            logger.info("Models loaded successfully")
    
    def get_word_embedding(self, word: str) -> np.ndarray:
        """Get word embedding from character-level autoencoder"""
        char_indices = [self.char2idx.get(c, self.char2idx['<PAD>']) for c in word]
        char_tensor = torch.tensor(char_indices, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedded = self.autoencoder.embedding(char_tensor)
            _, hidden = self.autoencoder.encoder_gru(embedded)
        
        word_emb = hidden[-1].squeeze(0)
        return word_emb.cpu().numpy()
    
    def predict_sentence(self, sentence: str) -> List[Dict[str, str]]:
        """Predict NER tags for a sentence"""
        # Ensure model is loaded
        if self.model is None:
            raise RuntimeError("CRF model not available. Install 'pytorch-crf' (pip install pytorch-crf) or 'torchcrf' and restart the server.")
        try:
            # Tokenize (simple whitespace tokenization for Khmer)
            tokens = sentence.strip().split()
            
            # Get embeddings for all tokens
            word_embs = []
            for token in tokens:
                emb = self.get_word_embedding(token)
                word_embs.append(emb)
            
            # Prepare input tensor
            X = torch.tensor([word_embs], dtype=torch.float32).to(self.device)
            mask = torch.ones(1, len(tokens), dtype=torch.bool).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                pred_ids = self.model.predict(X, mask)[0]
            
            # Convert to labels
            pred_labels = [self.idx2label[i] for i in pred_ids]
            
            # Format results
            results = []
            for token, label in zip(tokens, pred_labels):
                results.append({
                    "token": token,
                    "label": label,
                    "entity_type": self._get_entity_type(label),
                    "confidence": 0.95  # Could be calculated from model probabilities
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, sentences: List[str]) -> List[List[Dict[str, str]]]:
        """Predict NER tags for multiple sentences"""
        results = []
        for sentence in sentences:
            try:
                preds = self.predict_sentence(sentence)
                results.append(preds)
            except Exception as e:
                logger.error(f"Error processing sentence: {e}")
                results.append([])
        return results
    
    def _get_entity_type(self, label: str) -> str:
        """Extract entity type from BIO label"""
        if label == "O":
            return "O"
        return label.split("-")[1] if "-" in label else label