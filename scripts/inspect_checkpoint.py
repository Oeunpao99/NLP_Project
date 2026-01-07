import torch
p='ml/model/bilstm_crf_best.pt'
ckpt = torch.load(p, map_location='cpu')
if isinstance(ckpt, dict) and ('state_dict' in ckpt or 'model_state_dict' in ckpt):
    sd = ckpt.get('state_dict', ckpt.get('model_state_dict'))
elif isinstance(ckpt, dict):
    sd = ckpt
else:
    sd = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else None

if sd is None:
    print('state dict is None')
else:
    for k in sorted(sd.keys()):
        v = sd[k]
        print(f"{k} {tuple(v.shape)}")
