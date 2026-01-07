import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_service import CRF
import inspect
print('CRF init signature:', inspect.signature(CRF.__init__))
print('CRF doc sample:', (CRF.__doc__ or '')[:300])
print('\nCRF methods:')
print([n for n in dir(CRF) if not n.startswith('_')])
# Try to create and inspect methods
crf = CRF(5, pad_idx=None, use_gpu=False)
print('\nCRF instance created, callables:')
print([n for n in dir(crf) if callable(getattr(crf,n)) and not n.startswith('_')])
