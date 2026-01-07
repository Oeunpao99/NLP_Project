from app.core.config import settings
from app.models.database import engine
from sqlalchemy import text

db_name = settings.db_name
with engine.connect() as conn:
    res = conn.execute(text("SELECT COLUMN_NAME, DATA_TYPE FROM information_schema.columns WHERE table_schema = :db AND table_name = 'ner_predictions'"), {'db': db_name})
    cols = [(r.COLUMN_NAME, r.DATA_TYPE) for r in res]
    print('Columns:', cols)
