import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_service import CRF
import inspect
print('forward sig:', inspect.signature(CRF.forward))
print('viterbi_decode sig:', inspect.signature(CRF.viterbi_decode))
print('forward doc:', (CRF.forward.__doc__ or '')[:400])
print('viterbi doc:', (CRF.viterbi_decode.__doc__ or '')[:400])
