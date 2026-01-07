"""Initialize database and optionally create an admin user."""
import os
import sys
import logging

# Ensure the project root (/app) is on sys.path so `from app...` imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Also include current working dir and /app explicitly (covers different run contexts)
cwd = os.path.abspath(os.getcwd())
if cwd not in sys.path:
    sys.path.insert(0, cwd)
if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from app.models.database import init_db, SessionLocal
from app.models.user import User
from app.services.user_service import get_user_by_email, create_user

logger = logging.getLogger("init_db")


def main():
    try:
        logger.info("Initializing database...")
        init_db()
        # Optionally create admin user
        admin_email = os.getenv("ADMIN_EMAIL")
        admin_password = os.getenv("ADMIN_PASSWORD")
        db = SessionLocal()
        if admin_email and admin_password:
            if not get_user_by_email(db, admin_email):
                logger.info("Creating admin user")
                create_user(db, admin_email, admin_password)
            else:
                logger.info("Admin user already exists")
        db.close()
        logger.info("Database initialization complete")
    except Exception as e:
        logger.exception(f"Database initialization failed: {e}")


if __name__ == "__main__":
    main()