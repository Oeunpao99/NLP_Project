import json
from ml.ner_predictor import KhmerNERPredictor

p = KhmerNERPredictor('ml/model')
print('device:', p.device)
print('tagset_size:', p.tagset_size)
print('idx2label:', json.dumps({str(k): v for k, v in p.idx2label.items()}, ensure_ascii=False))
