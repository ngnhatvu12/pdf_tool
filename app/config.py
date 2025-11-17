from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:123@localhost:5433/aimdb")
    APP_ENV: str = os.getenv("APP_ENV", "dev")

settings = Settings()
