# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.5):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        
        # Embedding layer
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                           num_layers=num_layers, 
                           bidirectional=True, 
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        
        # Linear layer to map LSTM output to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
        # CRF layer
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))
        
        # Initialize transitions
        self.transitions.data[:, 0] = -10000  # Start tag constraints
        self.transitions.data[0, :] = -10000  # End tag constraints
        
        self.dropout = nn.Dropout(dropout)
        
    def _get_lstm_features(self, sentence, mask=None):
        """Get emission scores from LSTM"""
        embeds = self.word_embeds(sentence)
        embeds = self.dropout(embeds)
        
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
            packed_lstm_out, _ = self.lstm(packed_embeds)
            lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embeds)
        
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def _forward_alg(self, feats, mask):
        """Forward algorithm for CRF"""
        batch_size, seq_len, tag_size = feats.size()
        
        # Initialize alpha with the first step
        alpha = torch.full((batch_size, tag_size), -10000.0, device=feats.device)
        alpha[:, 0] = 0.0  # Start tag index is 0
        
        # Iterate through sequence
        for i in range(seq_len):
            # Get mask for current timestep
            mask_i = mask[:, i].unsqueeze(1)  # (batch_size, 1)
            
            # Expand alpha for broadcasting
            emit_score = feats[:, i].unsqueeze(1)  # (batch_size, 1, tag_size)
            
            # Compute scores for next step
            next_tag_var = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)  # (batch_size, tag_size, tag_size)
            next_tag_var = torch.logsumexp(next_tag_var, dim=1)  # (batch_size, tag_size)
            
            # Update alpha
            alpha = next_tag_var + emit_score.squeeze(1)
            
            # Apply mask
            alpha = alpha * mask_i + alpha * (1 - mask_i)
            
        # Add end transition
        terminal_var = alpha + self.transitions[0].unsqueeze(0)  # End tag index is 0
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha.sum()
    
    def _score_sentence(self, feats, tags, mask):
        """Score the given tag sequence"""
        batch_size, seq_len, tag_size = feats.size()
        
        # Convert tags to indices if they're one-hot
        if tags.dim() == 3:
            tags = tags.argmax(dim=2)
        
        score = torch.zeros(batch_size, device=feats.device)
        
        # Add start transition score
        start_tags = torch.full((batch_size, 1), 0, dtype=torch.long, device=feats.device)  # Start tag index = 0
        tags = torch.cat([start_tags, tags], dim=1)
        
        for i in range(seq_len):
            # Get mask for current timestep
            mask_i = mask[:, i]
            
            # Emission score
            emit_score = torch.gather(feats[:, i], 1, tags[:, i+1].unsqueeze(1)).squeeze(1)
            
            # Transition score
            trans_score = self.transitions[tags[:, i+1], tags[:, i]]
            
            # Add to score
            score += (emit_score + trans_score) * mask_i
        
        # Add end transition score
        last_tag = tags.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.transitions[0, last_tag]  # End tag index = 0
        
        return score.sum()
    
    def _viterbi_decode(self, feats, mask):
        """Viterbi decoding to find the best tag sequence"""
        batch_size, seq_len, tag_size = feats.size()
        
        # Initialize backpointers
        backpointers = []
        
        # Initialize viterbi variables
        viterbi = torch.full((batch_size, tag_size), -10000.0, device=feats.device)
        viterbi[:, 0] = 0  # Start tag index = 0
        
        # Forward pass
        for i in range(seq_len):
            mask_i = mask[:, i].unsqueeze(1)
            
            # Expand viterbi for broadcasting
            viterbi_expanded = viterbi.unsqueeze(2)  # (batch_size, tag_size, 1)
            trans_expanded = self.transitions.unsqueeze(0)  # (1, tag_size, tag_size)
            
            # Compute scores
            next_tag_var = viterbi_expanded + trans_expanded  # (batch_size, tag_size, tag_size)
            best_tag_scores, best_tag_ids = torch.max(next_tag_var, dim=1)
            
            # Store backpointers
            backpointers.append(best_tag_ids)
            
            # Update viterbi
            viterbi = best_tag_scores + feats[:, i]
            
            # Apply mask
            viterbi = viterbi * mask_i + viterbi * (1 - mask_i)
        
        # Add end transition
        terminal_var = viterbi + self.transitions[0].unsqueeze(0)
        best_tag_scores, best_tag_ids = torch.max(terminal_var, dim=1)
        
        # Backtrack to get best path
        best_paths = []
        for i in range(batch_size):
            # Follow backpointers
            best_tag = best_tag_ids[i]
            best_path = [best_tag.item()]
            
            for j in reversed(range(seq_len)):
                best_tag = backpointers[j][i, best_tag]
                best_path.insert(0, best_tag.item())
            
            # Remove start tag
            best_path = best_path[1:]
            best_paths.append(best_path)
        
        return best_paths, best_tag_scores
    
    def forward(self, sentence, mask=None):
        """Forward pass for inference"""
        lstm_feats = self._get_lstm_features(sentence, mask)
        
        if mask is None:
            mask = torch.ones_like(sentence, dtype=torch.uint8)
        
        # Viterbi decoding
        best_paths, scores = self._viterbi_decode(lstm_feats, mask)
        return best_paths
    
    def loss(self, sentence, tags, mask=None):
        """Compute negative log likelihood loss"""
        lstm_feats = self._get_lstm_features(sentence, mask)
        
        if mask is None:
            mask = torch.ones_like(sentence, dtype=torch.uint8)
        
        # Forward score
        forward_score = self._forward_alg(lstm_feats, mask)
        
        # Gold score
        gold_score = self._score_sentence(lstm_feats, tags, mask)
        
        return forward_score - gold_score