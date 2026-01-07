import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from .model_service import service

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
INDEX_HTML = os.path.join(ROOT_DIR, 'index1.html')

app = FastAPI(title='Khmer NER API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def index():
    if os.path.exists(INDEX_HTML):
        return FileResponse(INDEX_HTML, media_type='text/html')
    return JSONResponse({'message': 'index1.html not found'}, status_code=404)


@app.get('/api/status')
def status():
    return JSONResponse({'ready': service.ready})


@app.get('/api/debug')
def debug():
    # Returns detailed debug info about which files are present and what is loaded
    return JSONResponse(service.info())


@app.post('/api/reload')
def reload_model():
    # Force the service to re-scan model files and attempt to load
    service._try_load_defaults()
    return JSONResponse({'reloaded': True, 'info': service.info()})


@app.post('/api/predict')
def predict(payload: dict):
    text = payload.get('text', '') if isinstance(payload, dict) else ''
    if not text:
        raise HTTPException(status_code=400, detail='text required')
    out = service.predict(text)
    return JSONResponse(out)


@app.post('/api/upload')
async def upload(files: List[UploadFile] = File(...)):
    saved = {}
    allowed = {
        'bilstm': 'bilstm_crf_best.pt',
        'autoencoder': 'char_autoencoder_epoch3.pt',
        'idx2label': 'idx2label.json',
        'char2idx': 'char2idx.json'
    }
    file_map = {}
    for f in files:
        name = f.filename
        # try to infer key by filename or client-provided filename
        key = None
        if name in allowed.values():
            # get the logical key
            for k, v in allowed.items():
                if v == name:
                    key = k
                    break
        else:
            # if filename contains 'bilstm' etc
            lname = name.lower()
            for k in allowed.keys():
                if k in lname:
                    key = k
                    break
        if not key:
            # skip unknown
            continue
        content = await f.read()
        file_map[key] = content
    if not file_map:
        raise HTTPException(status_code=400, detail='No recognized files uploaded. Allowed: bilstm, autoencoder, idx2label, char2idx')
    saved_paths = service.load_uploaded_files(file_map)
    return JSONResponse({'saved': saved_paths, 'ready': service.ready})

