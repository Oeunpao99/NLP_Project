@echo off
REM Activate venv and run Uvicorn (Windows)
if not exist .venv (echo Create virtualenv with: python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt && exit /b)
.\.venv\Scripts\activate
uvicorn server.app:app --reload --port 8000
