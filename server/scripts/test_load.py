import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from server.model_service import BiLSTM_CRF
import torch
import traceback
s = torch.load('server/model/bilstm_crf_best.pt', map_location='cpu')
print('loaded object type:', type(s))
try:
    m = BiLSTM_CRF(embedding_dim=256, hidden_dim=128, tagset_size=11)
    m.load_state_dict(s)
    print('strict load succeeded')
except Exception as e:
    print('strict load failed:')
    traceback.print_exc()
try:
    m = BiLSTM_CRF(embedding_dim=256, hidden_dim=128, tagset_size=11)
    m.load_state_dict(s, strict=False)
    print('non-strict load succeeded')
except Exception as e:
    print('non-strict load failed:')
    traceback.print_exc()
