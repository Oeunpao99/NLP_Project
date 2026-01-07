"""
Database configuration for MySQL with SQLAlchemy
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Create MySQL database engine
# Format: mysql+pymysql://username:password@localhost/dbname
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=30,
    echo=False,
    connect_args={
        "connect_timeout": 10
    }
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

# Dependency to get database session
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables and apply lightweight schema updates"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")

        # Lightweight schema updates for backward compatibility (e.g., add new columns)
        with engine.connect() as conn:
            # Add user_id column if missing
            res = conn.execute(text("SELECT COUNT(*) as c FROM information_schema.columns WHERE table_schema = :db AND table_name = 'ner_predictions' AND column_name = 'user_id'"), {'db': settings.db_name})
            if res.fetchone().c == 0:
                logger.info("Adding missing column 'user_id' to ner_predictions table")
                conn.execute(text("ALTER TABLE ner_predictions ADD COLUMN user_id VARCHAR(36) NULL"))
                try:
                    conn.execute(text("CREATE INDEX ix_ner_predictions_user_id ON ner_predictions (user_id)"))
                except Exception:
                    # ignore index creation errors (e.g., if index already exists)
                    pass
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def drop_db():
    """Drop all database tables (for testing)"""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Error dropping database tables: {e}")
        raise