from app.core.config import settings
from app.models.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    # Check if user_id exists
    res = conn.execute(text("SELECT COUNT(*) as c FROM information_schema.columns WHERE table_schema = :db AND table_name = 'ner_predictions' AND column_name = 'user_id'"), {'db': settings.db_name})
    c = res.fetchone().c
    if c == 0:
        print('Adding user_id column...')
        conn.execute(text("ALTER TABLE ner_predictions ADD COLUMN user_id VARCHAR(36) NULL"))
        conn.execute(text("CREATE INDEX ix_ner_predictions_user_id ON ner_predictions (user_id)"))
        print('user_id column added')
    else:
        print('user_id column already exists')
