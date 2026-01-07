# Khmer NER - FastAPI backend

This folder contains a minimal FastAPI backend that integrates with the `index1.html` frontend in the workspace.

What it provides

- GET / -> serves `index1.html`
- GET /api/status -> returns `{ "ready": true|false }`
- POST /api/predict -> expects JSON `{ "text": "..." }`, returns `{ entities: [...] }` (uses model if loaded, otherwise heuristic fallback)
- POST /api/upload -> upload model files (multipart form). Accepted files: `bilstm_crf_best.pt` (or a file with `bilstm` in its name), `char_autoencoder.pt` (or `autoencoder` in name), `idx2label.json`, `char2idx.json`.

Quick start (development)

1. Create a virtualenv and install requirements:

   python -m venv .venv
   .\.venv\Scripts\activate (Windows)
   pip install -r server\requirements.txt

2. Run the server

   uvicorn server.app:app --reload --port 8000

3. Open http://127.0.0.1:8000 in your browser to load the frontend.

How to deploy your trained model

- Place your `bilstm_crf_best.pt` in `server/model/` and the autoencoder checkpoint `char_autoencoder.pt` (if you have it), and create correct `idx2label.json` and `char2idx.json` there. Restart the server.
- Alternatively, use `POST /api/upload` to send files from a client (multipart form upload).

Notes and caveats

- The service contains a heuristic fallback (regex-based) used when model files are missing or loading/prediction fails. This ensures the frontend continues working.
- The model-based predictor requires the `torchcrf` package. If `torchcrf` is not installed, the server will still start but will skip loading the BiLSTM+CRF model and fall back to heuristics. Install with:

  ```bash
  pip install torchcrf
  # or install all requirements:
  pip install -r server/requirements.txt
  ```

- For accurate model predictions you should provide the same `char2idx.json` and the char autoencoder checkpoint used during training, plus a correct mapping `idx2label.json` exported from your training environment.

If you want, I can add an automated script to export `idx2label` and `char2idx` from your notebook to JSON files — tell me and I will add it.
