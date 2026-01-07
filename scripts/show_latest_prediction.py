from app.models.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    res = conn.execute(text("SELECT id, text, inference_time_ms, user_id, metadata FROM ner_predictions ORDER BY created_at DESC LIMIT 1"))
    row = res.fetchone()
    print(row)
