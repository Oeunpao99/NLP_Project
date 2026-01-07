from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from typing import Optional

class Settings(BaseSettings):
    """Application settings with your specific ports"""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8001, alias="API_PORT")  # Your API port: 8001
    debug: bool = Field(default=False, alias="DEBUG")

    # Security / Auth
    secret_key: str = Field(default="change_me_to_a_secure_random_value", alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=60*24, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Database Configuration (legacy MySQL fields kept for backward compatibility)
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=3307, alias="DB_PORT")  # Your MySQL port: 3307
    db_name: str = Field(default="khmer_ner_db", alias="DB_NAME")
    db_user: str = Field(default="root", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")

    # Optional full database URL (e.g., a Postgres URL set by docker-compose)
    database_url_env: Optional[str] = Field(default=None, alias="DATABASE_URL")
    
    # ML Configuration
    model_dir: str = Field(default="ml/model", alias="MODEL_DIR")
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra='ignore'  # This prevents the "extra inputs" error
    )
    
    @property
    def database_url(self) -> str:
        """Return DATABASE_URL if provided, otherwise build a MySQL URL from components"""
        if self.database_url_env:
            return self.database_url_env
        password_part = f":{self.db_password}" if self.db_password else ""
        return (
            f"mysql+pymysql://{self.db_user}{password_part}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

settings = Settings()