from app.models.database import SessionLocal
import app.models.user  # ensure User mapper is registered
from app.models.ner_prediction import NERPrediction
import traceback

db = SessionLocal()
try:
    rec = NERPrediction(
        text='test insert',
        tokens=['a','b','c'],
        labels=['O','O','B-PER'],
        entities={'PER': ['a']},
        formatted_output='<p>test</p>',
        format='html',
        inference_time_ms=5.2,
        user_id=None,
        extra_metadata={'language':'khmer'}
    )
    db.add(rec)
    db.commit()
    print('inserted id:', rec.id)
except Exception:
    traceback.print_exc()
    db.rollback()
finally:
    db.close()
